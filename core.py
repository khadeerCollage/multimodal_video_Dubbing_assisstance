# ============================================================================
# Trendy 2026 Audio Processing - Google Colab GPU Version
# ============================================================================

import subprocess
import shutil
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable
import uuid
import torch
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================
class AudioProcessingError(Exception):
    """Base exception for audio processing errors"""
    pass

class FFmpegError(AudioProcessingError):
    """FFmpeg operation failed"""
    pass

class DemucsError(AudioProcessingError):
    """Demucs separation failed"""
    pass

class ValidationError(AudioProcessingError):
    """Input validation failed"""
    pass


# ============================================================================
# Main Audio Processor
# ============================================================================
class CinematicAudioProcessor:
    """
    Professional-grade audio processing for movie dubbing.
    Optimized for Google Colab GPU execution.
    
    Features:
    - GPU acceleration (auto-detects Colab GPU)
    - Robust error handling
    - soundfile backend (fixes torchcodec issues)
    - Progress callbacks
    """
    
    def __init__(self, quality_mode: str = "pro", fast_mode: bool = False):
        """
        Args:
            quality_mode: 
                - "basic": FFmpeg only (fast, no music preservation)
                - "pro": FFmpeg + Demucs (Netflix-quality)
            fast_mode: If True, use faster settings
        """
        self.quality_mode = quality_mode
        self.fast_mode = fast_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.job_id = str(uuid.uuid4())[:8]
        
        logger.info(f"AudioProcessor initialized: mode={quality_mode}, device={self.device}, fast={fast_mode}")
        
        # Check dependencies
        self._check_ffmpeg()
        if quality_mode != "basic":
            self._check_demucs()
    
    def _check_ffmpeg(self):
        """Verify FFmpeg is installed (Colab has it by default)"""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5, check=True)
            logger.info("✓ FFmpeg available")
        except Exception:
            raise ValidationError("FFmpeg not installed. Run: !apt-get install -y ffmpeg")
    
    def _check_demucs(self):
        """Verify Demucs and soundfile are installed"""
        try:
            subprocess.run(["demucs", "--help"], capture_output=True, timeout=5, check=True)
            logger.info("✓ Demucs available")
            
            # Check soundfile backend (CRITICAL for Colab)
            try:
                import soundfile
                logger.info("✓ soundfile backend available")
            except ImportError:
                logger.warning("soundfile not installed!")
                raise ValidationError(
                    "Missing soundfile. Run in Colab cell:\n"
                    "!pip install soundfile"
                )
            
            logger.info("Note: First run will download AI model (~2GB)")
            
        except subprocess.CalledProcessError:
            raise ValidationError("Demucs not installed. Run: !pip install demucs")
    
    def _validate_video(self, video_path: str):
        """Validate video file exists"""
        path = Path(video_path)
        if not path.exists():
            raise ValidationError(f"Video file not found: {video_path}")
        if not path.is_file():
            raise ValidationError(f"Not a file: {video_path}")
    
    def extract_raw_audio(
        self,
        video_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Stage 1: Extract raw audio from video (FFmpeg)
        """
        logger.info(f"[Stage 1] Extracting audio from: {Path(video_path).name}")
        
        if progress_callback:
            progress_callback("Extracting audio from video...")
        
        # Always use mono for pipeline consistency (required by Pyannote later)
        sample_rate = "22050" if self.fast_mode else "44100"
        
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", sample_rate,
            "-ac", "1",  # Mono (required for diarization)
            output_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300
            )
            
            if not Path(output_path).exists():
                raise FFmpegError("FFmpeg completed but output file not found")
            
            file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"[Stage 1] ✓ Extracted: {file_size_mb:.1f}MB (mono, {sample_rate}Hz)")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise FFmpegError(f"Audio extraction failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise FFmpegError("Audio extraction timeout")
    
    def separate_vocals(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, str]:
        """
        Stage 2: AI-powered stem separation with Demucs
        Uses soundfile backend to avoid torchcodec issues
        """
        if self.quality_mode == "basic":
            logger.info("[Stage 2] Skipped (basic mode)")
            return {"vocals": audio_path, "background": None}
        
        logger.info("[Stage 2] Separating stems with Demucs...")
        
        if progress_callback:
            progress_callback("Separating vocals from background (AI running)...")
        
        # Choose stable models
        model = "htdemucs_ft" if not self.fast_mode else "htdemucs"
        logger.info(f"Using model: {model}")
        
        # CRITICAL: Force soundfile backend (fixes torchcodec crash)
        env = dict(os.environ)
        env["TORCHAUDIO_USE_BACKEND"] = "soundfile"
        
        cmd = [
            "demucs",
            "--two-stems", "vocals",
            "-n", model,
            "--device", self.device,
            "-o", output_dir,
            audio_path
        ]
        
        if self.device == "cpu":
            cmd.extend(["-j", "4"])  # Use 4 cores on CPU
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,  # ← Use soundfile backend
                timeout=1800  # 30 min for long videos
            )
            
            if result.returncode != 0:
                logger.error("========== DEMUCS ERROR ==========")
                logger.error(result.stderr)
                raise DemucsError("Demucs failed. See logs above.")
            
            # Auto-discover output folder (handles any filename)
            model_dir = Path(output_dir) / model
            if not model_dir.exists():
                raise DemucsError(f"Output directory not found: {model_dir}")
            
            subfolders = [d for d in model_dir.iterdir() if d.is_dir()]
            if not subfolders:
                raise DemucsError("No Demucs output folder found")
            
            base_path = subfolders[0]
            
            # Find vocal and background files (any extension)
            vocals_files = list(base_path.glob("vocals.*"))
            bg_files = list(base_path.glob("no_vocals.*"))
            
            if not vocals_files:
                raise DemucsError(f"Vocals file not found in {base_path}")
            if not bg_files:
                raise DemucsError(f"Background file not found in {base_path}")
            
            vocals_path = str(vocals_files[0])
            bg_path = str(bg_files[0])
            
            logger.info(f"[Stage 2] ✓ Vocals: {Path(vocals_path).name}")
            logger.info(f"[Stage 2] ✓ Background: {Path(bg_path).name}")
            
            return {
                "vocals": vocals_path,
                "background": bg_path
            }
            
        except subprocess.TimeoutExpired:
            raise DemucsError("Demucs timeout (video too long)")
    
    def process_for_dubbing(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        cleanup_on_error: bool = True
    ) -> Tuple[str, Optional[str], Dict]:
        """
        Complete processing pipeline for dubbing workflow
        """
        self._validate_video(video_path)
        
        if output_dir is None:
            output_dir = f"dubbing_temp_{self.job_id}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing job: {self.job_id}")
        logger.info(f"Output directory: {output_dir}")
        
        try:
            # Stage 1: Extract audio
            raw_audio = str(output_path / "raw_audio.wav")
            self.extract_raw_audio(video_path, raw_audio, progress_callback)
            
            # Stage 2: Separate stems
            stems = self.separate_vocals(raw_audio, str(output_path), progress_callback)
            
            metadata = {
                "job_id": self.job_id,
                "video_source": video_path,
                "quality_mode": self.quality_mode,
                "fast_mode": self.fast_mode,
                "has_background_track": stems["background"] is not None,
                "device_used": self.device,
                "output_directory": str(output_path),
                "vocals_file": stems["vocals"],
                "background_file": stems["background"]
            }
            
            if progress_callback:
                progress_callback("Processing complete!")
            
            return stems["vocals"], stems["background"], metadata
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            
            if cleanup_on_error:
                logger.info("Cleaning up temporary files...")
                self.cleanup(output_dir)
            
            raise
    
    def remix_final_audio(
        self,
        dubbed_vocals: str,
        original_background: Optional[str],
        output_path: str,
        voice_volume: float = 1.0,
        bg_volume: float = 0.7
    ) -> str:
        """
        Stage 3: Mix dubbed voice with original background
        """
        logger.info("[Stage 3] Remixing final audio...")
        
        if not Path(dubbed_vocals).exists():
            raise ValidationError(f"Dubbed vocals not found: {dubbed_vocals}")
        
        if original_background and not Path(original_background).exists():
            raise ValidationError(f"Background track not found: {original_background}")
        
        try:
            if original_background is None:
                cmd = [
                    "ffmpeg", "-y", "-i", dubbed_vocals,
                    "-acodec", "aac", "-b:a", "192k",
                    output_path
                ]
            else:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", dubbed_vocals,
                    "-i", original_background,
                    "-filter_complex",
                    f"[0:a]volume={voice_volume}[voice];"
                    f"[1:a]volume={bg_volume}[bg];"
                    f"[voice][bg]amix=inputs=2:duration=first:dropout_transition=2",
                    "-acodec", "aac", "-b:a", "192k",
                    output_path
                ]
            
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            logger.info(f"[Stage 3] ✓ Final audio: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise FFmpegError(f"Audio mixing failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise FFmpegError("Audio mixing timeout")
    
    def cleanup(self, temp_dir: str):
        """Delete temporary files"""
        path = Path(temp_dir)
        if path.exists() and path.is_dir():
            try:
                shutil.rmtree(path)
                logger.info(f"✓ Cleaned up: {temp_dir}")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")


# ============================================================================
# Colab-Specific Test Runner
# ============================================================================

def test_colab_pipeline():
    """
    Test pipeline for Google Colab
    """
    import sys
    
    try:
        # Initialize processor
        processor = CinematicAudioProcessor(
            quality_mode="pro",
            fast_mode=True  # Faster for testing
        )
        
        # Get video file (from upload or argument)
        if len(sys.argv) > 1:
            video_file = sys.argv[1]
        else:
            video_file = "video.mp4"
        
        if not Path(video_file).exists():
            print("\n⚠ Video file not found!")
            print("In Colab, run this first:")
            print("  from google.colab import files")
            print("  uploaded = files.upload()")
            print("  video_file = list(uploaded.keys())[0]")
            return
        
        # Progress callback
        def progress(msg):
            print(f"  → {msg}")
        
        # Process video
        print("\n" + "="*60)
        print("Starting dubbing pipeline...")
        print("="*60 + "\n")
        
        vocals, background, metadata = processor.process_for_dubbing(
            video_file,
            progress_callback=progress,
            cleanup_on_error=True
        )
        
        print("\n" + "="*60)
        print("✅ Processing Complete!")
        print("="*60)
        print(f"\nVocals: {vocals}")
        print(f"Background: {background}")
        print(f"\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # Download results in Colab
        try:
            from google.colab import files
            print("\n📥 Downloading results...")
            files.download(vocals)
            if background:
                files.download(background)
            print("✓ Files downloaded!")
        except ImportError:
            print("\n(Not in Colab - files saved locally)")
        
        # Cleanup prompt
        cleanup = input("\nDelete temporary files? (y/n): ")
        if cleanup.lower() == 'y':
            processor.cleanup(metadata["output_directory"])
        
    except AudioProcessingError as e:
        print(f"\n❌ Error: {e}")
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_colab_pipeline()