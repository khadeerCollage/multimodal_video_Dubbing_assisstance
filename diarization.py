
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """Single speech segment with speaker identity and text"""
    speaker: str
    start: float
    end: float
    text: str
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass  
class TranscriptResult:
    """Complete transcript with metadata"""
    segments: List[SpeechSegment]
    num_speakers: int
    total_duration: float
    language: str
    
    def to_json(self, path: str):
        """Save transcript to JSON file"""
        data = {
            "num_speakers": self.num_speakers,
            "total_duration": self.total_duration,
            "language": self.language,
            "segments": [s.to_dict() for s in self.segments]
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved transcript to: {path}")
    
    @classmethod
    def from_json(cls, path: str) -> "TranscriptResult":
        """Load transcript from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        segments = [SpeechSegment(**s) for s in data["segments"]]
        return cls(
            segments=segments,
            num_speakers=data["num_speakers"],
            total_duration=data["total_duration"],
            language=data["language"]
        )


class SimpleSpeakerDiarizer:
    """
    Lightweight speaker diarization using audio features + clustering.
    No Pyannote, no HuggingFace token, no PyTorch Lightning.
    
    Approach:
    1. Extract audio segments from Whisper
    2. Compute simple audio features (energy, pitch) per segment
    3. Cluster similar segments → assign speaker IDs
    """
    
    def __init__(self, num_speakers: int = None, max_speakers: int = 5):
        self.num_speakers = num_speakers
        self.max_speakers = max_speakers
    
    def _extract_features(self, audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
        """Extract simple audio features for a segment"""
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = audio[start_sample:end_sample]
        
        if len(segment) < 100:
            return np.zeros(6)
        
        # Simple features: energy stats + zero crossing rate
        features = [
            np.mean(np.abs(segment)),           # Mean energy
            np.std(segment),                     # Energy variance
            np.max(np.abs(segment)),            # Peak energy
            np.sum(np.diff(np.sign(segment)) != 0) / len(segment),  # Zero crossing rate
            np.percentile(np.abs(segment), 75), # 75th percentile energy
            np.percentile(np.abs(segment), 25), # 25th percentile energy
        ]
        return np.array(features)
    
    def assign_speakers(
        self, 
        audio_path: str, 
        segments: List[Dict]
    ) -> List[str]:
        """
        Assign speaker IDs to transcribed segments using clustering.
        
        Args:
            audio_path: Path to audio file
            segments: List of segments with 'start' and 'end' times
            
        Returns:
            List of speaker IDs matching each segment
        """
        import soundfile as sf
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Analyzing speaker patterns...")
        
        # Load audio
        audio, sr = sf.read(audio_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        
        # Extract features for each segment
        features = []
        valid_indices = []
        
        for i, seg in enumerate(segments):
            feat = self._extract_features(audio, sr, seg['start'], seg['end'])
            if np.any(feat):  # Skip empty features
                features.append(feat)
                valid_indices.append(i)
        
        if len(features) < 2:
            # Not enough segments, assign all to SPEAKER_00
            return ["SPEAKER_00"] * len(segments)
        
        # Normalize features
        features = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Determine number of speakers
        n_clusters = self.num_speakers
        if n_clusters is None:
            # Auto-detect: use silhouette score
            from sklearn.metrics import silhouette_score
            best_score = -1
            best_k = 2
            
            for k in range(2, min(self.max_speakers + 1, len(features))):
                try:
                    clustering = AgglomerativeClustering(n_clusters=k)
                    labels = clustering.fit_predict(features_scaled)
                    score = silhouette_score(features_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                except:
                    continue
            
            n_clusters = best_k
        
        # Final clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(features_scaled)
        
        # Map back to all segments
        speaker_ids = ["SPEAKER_00"] * len(segments)
        for idx, label in zip(valid_indices, labels):
            speaker_ids[idx] = f"SPEAKER_{label:02d}"
        
        unique_speakers = set(speaker_ids)
        logger.info(f"✓ Detected {len(unique_speakers)} speakers")
        
        return speaker_ids

class TranscriptionEngine:
    """
    Speech-to-Text using Faster-Whisper (CTranslate2 optimized)
    
    Why Faster-Whisper?
    - 4x faster than OpenAI Whisper
    - 50% less memory
    - Better word-level timestamps
    - CPU optimized (int8)
    """
    
    def __init__(self, model_size: str = "large-v3"):
        """
        Args:
            model_size: tiny, base, small, medium, large-v2, large-v3
        """
        self.model_size = model_size
        self.model = None
    
    def _load_model(self):
        """Lazy load Faster-Whisper model"""
        if self.model is None:
            logger.info(f"Loading Faster-Whisper {self.model_size}...")
            
            from faster_whisper import WhisperModel
            import torch
            
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
            else:
                device = "cpu"
                compute_type = "int8"
            
            self.model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type
            )
            
            logger.info(f"✓ Loaded on {device} ({compute_type})")
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """
        Transcribe audio with word-level timestamps.
        """
        self._load_model()
        
        logger.info(f"Transcribing: {Path(audio_path).name}")
        
        segments_gen, info = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400
            )
        )
        
        segments = []
        for segment in segments_gen:
            seg_dict = {
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'words': []
            }
            
            if segment.words:
                for word in segment.words:
                    seg_dict['words'].append({
                        'word': word.word,
                        'start': word.start,
                        'end': word.end,
                        'probability': word.probability
                    })
            
            segments.append(seg_dict)
        
        logger.info(f"✓ {len(segments)} segments, language={info.language} ({info.language_probability:.0%})")
        
        return {
            'segments': segments,
            'language': info.language,
            'language_probability': info.language_probability
        }


class SpeechProcessor:
    """
    Complete Speech Processing Pipeline
    vocals.wav → Structured Transcript with Speaker IDs
    
    No Pyannote, no HuggingFace token, no dependency issues.
    """
    
    def __init__(self, whisper_model: str = "medium", num_speakers: int = None):
        """
        Args:
            whisper_model: Whisper model size (tiny/base/small/medium/large-v3)
            num_speakers: Number of speakers (None = auto-detect)
        """
        self.transcriber = TranscriptionEngine(whisper_model)
        self.diarizer = SimpleSpeakerDiarizer(num_speakers=num_speakers)
    
    def process(
        self,
        vocals_path: str,
        output_dir: Optional[str] = None,
        language: Optional[str] = None,
        max_speakers: int = 5
    ) -> TranscriptResult:
        """
        Complete processing: Transcription + Speaker Assignment
        """
        logger.info("=" * 60)
        logger.info(" Speech Processing Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Transcription
        logger.info("\n[Step 1/2] Transcribing speech...")
        whisper_result = self.transcriber.transcribe(vocals_path, language=language)
        
        if not whisper_result['segments']:
            logger.warning("No speech detected!")
            return TranscriptResult([], 0, 0.0, whisper_result.get('language', 'unknown'))
        
        # Step 2: Speaker Assignment
        logger.info("\n[Step 2/2] Identifying speakers...")
        self.diarizer.max_speakers = max_speakers
        speaker_ids = self.diarizer.assign_speakers(vocals_path, whisper_result['segments'])
        
        # Build final segments
        aligned_segments = []
        for seg, speaker in zip(whisper_result['segments'], speaker_ids):
            if seg['text']:
                aligned_segments.append(SpeechSegment(
                    speaker=speaker,
                    start=round(seg['start'], 3),
                    end=round(seg['end'], 3),
                    text=seg['text'],
                    confidence=1.0
                ))
        
        # Create result
        unique_speakers = set(s.speaker for s in aligned_segments)
        total_duration = max(s.end for s in aligned_segments) if aligned_segments else 0.0
        
        result = TranscriptResult(
            segments=aligned_segments,
            num_speakers=len(unique_speakers),
            total_duration=total_duration,
            language=whisper_result['language']
        )
        
        # Save if output_dir provided
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            # JSON
            result.to_json(str(out_path / "transcript.json"))
            
            # Human-readable
            txt_path = out_path / "transcript.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Transcript - {result.num_speakers} speakers\n")
                f.write(f"Duration: {result.total_duration:.1f}s | Language: {result.language}\n")
                f.write("=" * 60 + "\n\n")
                
                for seg in result.segments:
                    f.write(f"[{seg.start:6.1f}s → {seg.end:6.1f}s] {seg.speaker}:\n")
                    f.write(f"    \"{seg.text}\"\n\n")
            
            logger.info(f"✓ Saved: {txt_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info(" Processing Complete!")
        logger.info(f"   {len(aligned_segments)} segments, {len(unique_speakers)} speakers")
        logger.info("=" * 60)
        
        return result

def main():
    import sys
    
    print("\n" + "=" * 70)
    print(" Speech Processor - Transcription + Speaker Detection")
    print("=" * 70 + "\n")
    
    # Get vocals path
    if len(sys.argv) > 1:
        vocals_path = sys.argv[1]
    else:
        vocals_path = "dubbing_output_8ca37531/htdemucs/raw_audio/vocals.wav"
    
    if not Path(vocals_path).exists():
        print(f" File not found: {vocals_path}")
        print("\nUsage: python diarization.py <vocals.wav>")
        return
    
    try:
        # Process
        processor = SpeechProcessor(
            whisper_model="medium",  # Options: tiny, base, small, medium, large-v3
            num_speakers=None        # Auto-detect
        )
        
        output_dir = str(Path(vocals_path).parent.parent.parent / "transcript")
        
        transcript = processor.process(
            vocals_path=vocals_path,
            output_dir=output_dir,
            max_speakers=5
        )
        
        # Print sample
        print("\n Sample Output:")
        print("-" * 50)
        for seg in transcript.segments[:5]:
            print(f"[{seg.start:.1f}s → {seg.end:.1f}s] {seg.speaker}:")
            print(f"    \"{seg.text}\"\n")
        
        if len(transcript.segments) > 5:
            print(f"  ... and {len(transcript.segments) - 5} more segments")
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


