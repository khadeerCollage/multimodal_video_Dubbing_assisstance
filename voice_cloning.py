# ============================================================================
# Stage 5: TTS Voice Cloning Engine - Production Grade 2026
# English Audio Generation with Speaker-Aware Voice Cloning
# ============================================================================
#
# STACK:
# - Coqui XTTS v2 (Voice Cloning)
# - soundfile (Audio I/O - stable, no torchaudio crashes)
# - librosa (Time stretching for timing adjustment)
# - pydub (Audio assembly and mixing)
# ============================================================================

import os
import json
import logging
import soundfile as sf
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================
class VoiceCloningError(Exception):
    """Base exception for voice cloning errors"""
    pass

class TTSError(VoiceCloningError):
    """TTS generation failed"""
    pass

class AudioProcessingError(VoiceCloningError):
    """Audio processing/timing adjustment failed"""
    pass

class ReferenceExtractionError(VoiceCloningError):
    """Failed to extract reference audio"""
    pass


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class VoiceCloningConfig:
    """Configuration for voice cloning engine"""
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    language: str = "en"
    device: str = "cuda"  # or "cpu"
    sample_rate: int = 48000  # ← CHANGED from 22050
    speed_adjustment_threshold: float = 0.10  # 10% tolerance
    max_stretch_ratio: float = 0.15  # 15% max stretch
    emotion_prefix_enabled: bool = True
    checkpoint_every: int = 50  # Checkpoint every N segments
    
    # NEW: Audio enhancement
    enable_denoising: bool = True
    enable_compression: bool = True
    enable_crossfade: bool = True
    crossfade_duration: float = 0.05  # 50ms
    
    def __post_init__(self):
        # Auto-detect GPU
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"


# ============================================================================
# Voice Profile Manager
# ============================================================================
class VoiceProfileManager:
    """
    Extracts and manages speaker reference audio for voice cloning.
    Finds the best segments for each speaker from vocals.wav.
    """
    
    def __init__(self, config: VoiceCloningConfig):
        self.config = config
        self.profiles = {}
    
    def extract_reference(
        self, 
        speaker_id: str, 
        vocals_path: str, 
        segments: List[Dict],
        output_dir: str = "voice_references"
    ) -> str:
        """
        Extract best reference audio for speaker.
        
        Strategy:
        1. Find segments for this speaker
        2. Filter: confidence > 0.95, duration > 2.0s, neutral emotion
        3. Take top 3 longest segments
        4. Concatenate to 6-10 seconds
        """
        
        logger.info(f"Extracting reference for {speaker_id}...")
        
        # Filter for HIGH-QUALITY segments only
        speaker_segments = [
            seg for seg in segments 
            if seg.get("speaker") == speaker_id 
            and seg.get("duration", 0) > 3.0  # Longer = smoother
            and seg.get("emotion", "neutral") == "neutral"  # ONLY neutral
            and not seg.get("timing_valid", True) == False  # Avoid overflow segments
        ]
        
        if not speaker_segments:
            raise ReferenceExtractionError(f"No suitable segments found for {speaker_id}")
        
        # Sort by duration (longest first)
        speaker_segments.sort(key=lambda x: x.get("duration", 0), reverse=True)
        
        # Take top 3 segments
        best_segments = speaker_segments[:3]
        
        logger.info(f"   Found {len(best_segments)} reference segments:")
        for seg in best_segments:
            logger.info(f"      {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.2f}s)")
        
        # Load vocals.wav
        audio_data, sr = sf.read(vocals_path)
        
        # Extract and concatenate segments
        reference_audio = []
        for seg in best_segments:
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            segment_audio = audio_data[start_sample:end_sample]
            reference_audio.append(segment_audio)
        
        # Concatenate
        reference_audio = np.concatenate(reference_audio)
        
        # Ensure 6-10 seconds
        target_samples = int(12 * sr)  # 12 seconds ideal
        if len(reference_audio) > target_samples:
            reference_audio = reference_audio[:target_samples]
        
        # Save reference
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{speaker_id}_neutral.wav")
        sf.write(output_path, reference_audio, sr)
        
        duration = len(reference_audio) / sr
        logger.info(f"   ✓ Saved: {output_path} ({duration:.2f}s)")
        
        self.profiles[speaker_id] = output_path
        return output_path
    
    def get_reference(self, speaker_id: str) -> str:
        """Get reference audio path for speaker"""
        if speaker_id not in self.profiles:
            raise ReferenceExtractionError(f"No profile found for {speaker_id}")
        return self.profiles[speaker_id]


# ============================================================================
# TTS Engine (Coqui XTTS v2)
# ============================================================================
class TTSEngine:
    """
    Coqui XTTS v2 wrapper with timing adjustment.
    Generates English audio with voice cloning.
    """
    
    def __init__(self, config: VoiceCloningConfig):
        self.config = config
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize Coqui XTTS v2 model"""
        try:
            from TTS.api import TTS
            logger.info("Loading Coqui XTTS v2 model...")
            self.model = TTS(self.config.model_name).to(self.config.device)
            logger.info(f"✓ Model loaded on {self.config.device}")
        except ImportError:
            logger.error("TTS package not found. Install: pip install TTS")
            raise TTSError("TTS package not installed")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise TTSError(f"Model initialization failed: {e}")
    
    def generate(
        self, 
        text: str, 
        speaker_wav: str, 
        target_duration: Optional[float] = None,
        emotion: str = "neutral"
    ) -> np.ndarray:
        """
        Generate TTS audio with voice cloning.
        
        Args:
            text: English text to synthesize
            speaker_wav: Path to reference audio
            target_duration: Target duration in seconds (for timing adjustment)
            emotion: Emotion tag (for text prefix)
        
        Returns:
            Audio data as numpy array
        """
        
        if not self.model:
            raise TTSError("Model not initialized")
        
        # Emotion prefix (optional)
        if self.config.emotion_prefix_enabled and emotion != "neutral":
            emotion_map = {
                "anger": "[Angry] ",
                "sorrow": "[Sad] ",
                "fear": "[Fearful] ",
                "love": "[Lovingly] ",
                "shock": "[Shocked!] "
            }
            text = emotion_map.get(emotion, "") + text
        
        # Generate TTS
        try:
            # XTTS v2 generates to temp file
            temp_output = "temp_tts_output.wav"
            self.model.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=self.config.language,
                file_path=temp_output
            )
            
            # Load generated audio
            audio_data, sr = sf.read(temp_output)
            
            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
            
            # Resample if needed
            if sr != self.config.sample_rate:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.config.sample_rate)
            
            # Timing adjustment
            if target_duration:
                audio_data = self._adjust_timing(audio_data, target_duration)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise TTSError(f"Generation failed: {e}")
    
    def _adjust_timing(self, audio: np.ndarray, target_duration: float) -> np.ndarray:
        """
        Adjust audio duration to match target using speed/stretch.
        
        Strategy:
        - <10% difference: Speed adjustment (preserves pitch)
        - 10-15% difference: Time stretching (slight pitch change)
        - >15% difference: Flag for review, use stretch anyway
        """
        
        actual_duration = len(audio) / self.config.sample_rate
        ratio = target_duration / actual_duration
        
        if 0.999 <= ratio <= 1.001:  # Already perfect
            return audio
        
        try:
            import librosa
            
            if abs(1 - ratio) <= self.config.speed_adjustment_threshold:
                # Speed adjustment (natural)
                adjusted = librosa.effects.time_stretch(audio, rate=1/ratio)
            elif abs(1 - ratio) <= self.config.max_stretch_ratio:
                # Time stretching (acceptable)
                adjusted = librosa.effects.time_stretch(audio, rate=1/ratio)
            else:
                # Too much difference, flag but still stretch
                logger.warning(f"Large timing mismatch: ratio={ratio:.3f} (target={target_duration:.2f}s, actual={actual_duration:.2f}s)")
                adjusted = librosa.effects.time_stretch(
                    audio, 
                    rate=1/ratio,
                    n_fft=2048,  # Larger FFT = smoother
                    hop_length=512
                )
            
            return adjusted
            
        except Exception as e:
            logger.warning(f"Timing adjustment failed: {e}, using original")
            return audio


# ============================================================================
# Audio Assembler
# ============================================================================
class AudioAssembler:
    """
    Assembles individual segment audio files into final timeline.
    Handles silence insertion and background music mixing.
    """
    
    def __init__(self, config: VoiceCloningConfig):
        self.config = config
    
    def create_timeline(
        self, 
        segment_files: List[str], 
        segments_metadata: List[Dict],
        output_path: str
    ) -> str:
        """
        Create timeline audio from segments.
        Inserts silence gaps to match original timestamps.
        
        Args:
            segment_files: List of paths to segment .wav files
            segments_metadata: Original segment metadata with timestamps
            output_path: Output path for timeline audio
        
        Returns:
            Path to assembled audio
        """
        
        logger.info("Assembling audio timeline...")
        
        total_duration = segments_metadata[-1]['end']
        total_samples = int(total_duration * self.config.sample_rate)
        
        # Create silent timeline
        timeline = np.zeros(total_samples, dtype=np.float32)
        
        # Insert each segment at correct timestamp
        for seg_file, seg_meta in zip(segment_files, segments_metadata):
            # Load segment audio
            seg_audio, sr = sf.read(seg_file)
            
            # Resample if needed
            if sr != self.config.sample_rate:
                import librosa
                seg_audio = librosa.resample(seg_audio, orig_sr=sr, target_sr=self.config.sample_rate)
            
            # Calculate position in timeline
            start_sample = int(seg_meta['start'] * self.config.sample_rate)
            end_sample = start_sample + len(seg_audio)
            
            # Insert (handle overflow)
            if end_sample > len(timeline):
                end_sample = len(timeline)
                seg_audio = seg_audio[:end_sample - start_sample]
            
            timeline[start_sample:end_sample] = seg_audio
        
        # Save timeline
        sf.write(output_path, timeline, self.config.sample_rate)
        logger.info(f"   ✓ Timeline saved: {output_path} ({total_duration:.2f}s)")
        
        return output_path
    
    def mix_with_background(
        self, 
        vocals_path: str, 
        background_path: str, 
        output_path: str,
        vocals_volume: float = 1.0,
        background_volume: float = 0.7
    ) -> str:
        """
        Mix English vocals with original background music/SFX.
        Uses FFmpeg for professional mixing.
        """
        
        logger.info("Mixing with background audio...")
        
        import subprocess
        
        # FFmpeg command for mixing
        # [0] = vocals, [1] = background
        # amix: average, weights for volume control
        cmd = [
            'ffmpeg', '-y',
            '-i', vocals_path,
            '-i', background_path,
            '-filter_complex',
            f'[0:a]volume={vocals_volume}[a1];[1:a]volume={background_volume}[a2];[a1][a2]amix=inputs=2:duration=first',
            '-ar', str(self.config.sample_rate),
            '-ac', '2',  # Stereo
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"   ✓ Mixed audio saved: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg mixing failed: {e.stderr}")
            raise AudioProcessingError(f"Mixing failed: {e}")


# ============================================================================
# Main Voice Cloning Pipeline
# ============================================================================
class VoiceCloningPipeline:
    """
    Production-grade voice cloning pipeline.
    
    Features:
    - Speaker-aware voice cloning
    - Emotion-based text formatting
    - Timing adjustment for lip-sync
    - Checkpoint system for recovery
    - Progress tracking
    """
    
    def __init__(self, config: Optional[VoiceCloningConfig] = None):
        self.config = config or VoiceCloningConfig()
        self.profile_manager = VoiceProfileManager(self.config)
        self.tts_engine = TTSEngine(self.config)
        self.assembler = AudioAssembler(self.config)
    
    def load_translated_json(self, path: str) -> Dict:
        """Load translated.json"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_checkpoint(self, path: str, data: Dict):
        """Save checkpoint with atomic write"""
        checkpoint_path = path + ".checkpoint"
        try:
            temp_path = checkpoint_path + ".tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            shutil.move(temp_path, checkpoint_path)
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
    
    def load_checkpoint(self, path: str) -> Optional[Dict]:
        """Load checkpoint if exists"""
        checkpoint_path = path + ".checkpoint"
        if Path(checkpoint_path).exists():
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"   📂 Checkpoint found: {data['processed_segments']} segments done")
                    return data
            except Exception as e:
                logger.warning(f"   Corrupted checkpoint: {e}")
                return None
        return None
    
    def process(
        self,
        translated_json_path: str,
        vocals_wav_path: str,
        no_vocals_wav_path: str,
        output_dir: str = "audio_segments",
        resume: bool = True
    ) -> Dict:
        """
        Main voice cloning pipeline.
        
        Args:
            translated_json_path: Path to translated.json
            vocals_wav_path: Path to vocals.wav (for reference extraction)
            no_vocals_wav_path: Path to no_vocals.wav (background)
            output_dir: Output directory for segment audio files
            resume: Resume from checkpoint if exists
        
        Returns:
            Result dictionary with stats
        """
        
        logger.info("=" * 70)
        logger.info(" Stage 5: Voice Cloning Pipeline (Coqui XTTS v2)")
        logger.info(f"   Device: {self.config.device}")
        logger.info(f"   Language: {self.config.language}")
        logger.info("=" * 70)
        
        # Load translated data
        logger.info(f"\n[1/5] Loading translated segments...")
        translated_data = self.load_translated_json(translated_json_path)
        segments = translated_data.get("segments", [])
        
        if not segments:
            logger.error("No segments found!")
            return {"error": "no_segments"}
        
        logger.info(f"   Found {len(segments)} segments")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load checkpoint
        processed_files = []
        start_idx = 0
        
        if resume:
            checkpoint = self.load_checkpoint(os.path.join(output_dir, "voice_cloning"))
            if checkpoint:
                processed_files = checkpoint.get("processed_files", [])
                start_idx = checkpoint.get("processed_segments", 0)
                logger.info(f"   ✅ Resuming from segment {start_idx}")
        
        # Step 2: Extract voice references
        logger.info(f"\n[2/5] Extracting voice references...")
        speakers = set(seg.get("speaker") for seg in segments)
        
        for speaker in speakers:
            try:
                self.profile_manager.extract_reference(
                    speaker, 
                    vocals_wav_path, 
                    segments
                )
            except Exception as e:
                logger.error(f"Failed to extract reference for {speaker}: {e}")
                return {"error": f"reference_extraction_failed: {e}"}
        
        # Step 3: Generate TTS audio for each segment
        logger.info(f"\n[3/5] Generating TTS audio ({len(segments)} segments)...")
        
        stats = {
            "total_segments": len(segments),
            "processed": len(processed_files),
            "failed": 0,
            "timing_adjusted": 0
        }
        
        try:
            for idx, segment in enumerate(segments):
                if idx < start_idx:
                    continue
                
                # Progress
                progress = (idx + 1) / len(segments) * 100
                logger.info(f"   [{idx+1}/{len(segments)}] ({progress:.1f}%) - {segment.get('speaker')}: \"{segment.get('translated', '')[:50]}...\"")
                
                # Generate TTS
                try:
                    speaker_ref = self.profile_manager.get_reference(segment.get("speaker"))
                    
                    audio_data = self.tts_engine.generate(
                        text=segment.get("translated", ""),
                        speaker_wav=speaker_ref,
                        target_duration=segment.get("duration"),
                        emotion=segment.get("emotion", "neutral")
                    )
                    
                    # Save segment
                    segment_file = os.path.join(output_dir, f"segment_{idx:04d}.wav")
                    sf.write(segment_file, audio_data, self.config.sample_rate)
                    processed_files.append(segment_file)
                    
                    stats["processed"] += 1
                    
                    # Check if timing was adjusted
                    actual_duration = len(audio_data) / self.config.sample_rate
                    if abs(actual_duration - segment.get("duration", 0)) > 0.1:
                        stats["timing_adjusted"] += 1
                    
                except Exception as e:
                    logger.error(f"      Failed: {e}")
                    stats["failed"] += 1
                    # Create silent segment as fallback
                    silent_audio = np.zeros(int(segment.get("duration", 1.0) * self.config.sample_rate))
                    segment_file = os.path.join(output_dir, f"segment_{idx:04d}.wav")
                    sf.write(segment_file, silent_audio, self.config.sample_rate)
                    processed_files.append(segment_file)
                
                # Checkpoint every N segments
                if (idx + 1) % self.config.checkpoint_every == 0:
                    checkpoint_data = {
                        "processed_segments": idx + 1,
                        "processed_files": processed_files,
                        "stats": stats,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.save_checkpoint(os.path.join(output_dir, "voice_cloning"), checkpoint_data)
                    logger.info(f"      💾 Checkpoint saved")
        
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Interrupted!")
            checkpoint_data = {
                "processed_segments": idx,
                "processed_files": processed_files,
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            self.save_checkpoint(os.path.join(output_dir, "voice_cloning"), checkpoint_data)
            return {"interrupted": True, "stats": stats}
        
        # Step 4: Assemble timeline
        logger.info(f"\n[4/5] Assembling audio timeline...")
        timeline_path = os.path.join(output_dir, "english_vocals.wav")
        self.assembler.create_timeline(processed_files, segments, timeline_path)
        
        # Step 5: Mix with background
        logger.info(f"\n[5/5] Mixing with background audio...")
        final_audio_path = "final_english_audio.wav"
        self.assembler.mix_with_background(
            timeline_path,
            no_vocals_wav_path,
            final_audio_path,
            vocals_volume=1.0,
            background_volume=0.65
        )
        
        # Cleanup checkpoint
        checkpoint_path = os.path.join(output_dir, "voice_cloning.checkpoint")
        if Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()
            logger.info("   🗑️ Checkpoint cleaned")
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info(" Voice Cloning Complete!")
        logger.info(f"   Segments: {stats['total_segments']}")
        logger.info(f"   Processed: {stats['processed']}")
        logger.info(f"   Failed: {stats['failed']}")
        logger.info(f"   Timing Adjusted: {stats['timing_adjusted']}")
        logger.info(f"   Output: {final_audio_path}")
        logger.info("=" * 70)
        
        return {
            "success": True,
            "stats": stats,
            "output_audio": final_audio_path,
            "segments_dir": output_dir
        }


# ============================================================================
# CLI Entry Point
# ============================================================================
def main():
    import sys
    
    print("\n" + "=" * 70)
    print(" Stage 5: Voice Cloning Engine (Coqui XTTS v2)")
    print("   English Audio Generation with Speaker Cloning")
    print("=" * 70 + "\n")
    
    translated_json = sys.argv[1] if len(sys.argv) > 1 else "dubbing_output_8ca37531/transcript/translated.json"
    vocals_wav = sys.argv[2] if len(sys.argv) > 2 else "dubbing_output_8ca37531/vocals.wav"
    no_vocals_wav = sys.argv[3] if len(sys.argv) > 3 else "dubbing_output_8ca37531/no_vocals.wav"
    
    # Validate inputs
    for path, name in [(translated_json, "translated.json"), (vocals_wav, "vocals.wav"), (no_vocals_wav, "no_vocals.wav")]:
        if not Path(path).exists():
            print(f"❌ Not found: {path}")
            print(f"\nUsage: python voice_cloning.py <translated.json> <vocals.wav> <no_vocals.wav>")
            return
    
    print(f"✓ Input files validated")
    print(f"   Translated: {translated_json}")
    print(f"   Vocals: {vocals_wav}")
    print(f"   Background: {no_vocals_wav}\n")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        config = VoiceCloningConfig(device="cuda")
    else:
        print("⚠️ No GPU detected, using CPU (will be slower)")
        config = VoiceCloningConfig(device="cpu")
    
    try:
        pipeline = VoiceCloningPipeline(config)
        result = pipeline.process(
            translated_json_path=translated_json,
            vocals_wav_path=vocals_wav,
            no_vocals_wav_path=no_vocals_wav
        )
        
        if result.get("success"):
            print(f"\n✅ SUCCESS! Final audio: {result['output_audio']}")
            print(f"   Ready for Stage 6 (Lip-Sync)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted! Progress saved, resume with same command.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()