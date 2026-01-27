# # ============================================================================
# # Trendy 2026 Audio Processing - Google Colab GPU Version
# # FIXED: Mandatory Pro Mode (Demucs Always Enabled)
# # ============================================================================

# import subprocess
# import shutil
# import logging
# from pathlib import Path
# from typing import Dict, Tuple, Optional, Callable
# import uuid
# import torch
# import os

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
# logger = logging.getLogger(__name__)


# # ============================================================================
# # Custom Exceptions
# # ============================================================================
# class AudioProcessingError(Exception):
#     """Base exception for audio processing errors"""
#     pass

# class FFmpegError(AudioProcessingError):
#     """FFmpeg operation failed"""
#     pass

# class DemucsError(AudioProcessingError):
#     """Demucs separation failed"""
#     pass

# class ValidationError(AudioProcessingError):
#     """Input validation failed"""
#     pass


# # ============================================================================
# # Main Audio Processor
# # ============================================================================
# class CinematicAudioProcessor:
#     """
#     Professional-grade audio processing for movie dubbing.
#     Optimized for Google Colab GPU execution.
    
#     Features:
#     - GPU acceleration (auto-detects Colab GPU)
#     - Robust error handling
#     - soundfile backend (fixes torchcodec issues)
#     - Progress callbacks
#     - PRO MODE ONLY: Demucs stem separation always enabled
#     """
    
#     def __init__(self, fast_mode: bool = False):
#         """
#         Args:
#             fast_mode: If True, use faster model (htdemucs instead of htdemucs_ft)
        
#         Note: Pro mode (Demucs stem separation) is ALWAYS enabled
#         """
#         self.quality_mode = "pro"  # ← MANDATORY PRO MODE
#         self.fast_mode = fast_mode
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.job_id = str(uuid.uuid4())[:8]
        
#         logger.info(f"AudioProcessor initialized: mode=PRO (mandatory), device={self.device}, fast={fast_mode}")
        
#         # Check dependencies
#         self._check_ffmpeg()
#         self._check_demucs()  # ← Always check Demucs
    
#     def _check_ffmpeg(self):
#         """Verify FFmpeg is installed (Colab has it by default)"""
#         try:
#             subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5, check=True)
#             logger.info("✓ FFmpeg available")
#         except Exception:
#             raise ValidationError("FFmpeg not installed. Run: !apt-get install -y ffmpeg")
    
#     def _check_demucs(self):
#         """Verify Demucs and soundfile are installed"""
#         try:
#             subprocess.run(["demucs", "--help"], capture_output=True, timeout=5, check=True)
#             logger.info("✓ Demucs available")
            
#             # Check soundfile backend (CRITICAL for Colab)
#             try:
#                 import soundfile
#                 logger.info("✓ soundfile backend available")
#             except ImportError:
#                 logger.warning("soundfile not installed!")
#                 raise ValidationError(
#                     "Missing soundfile. Run in Colab cell:\n"
#                     "!pip install soundfile"
#                 )
            
#             logger.info("Note: First run will download AI model (~2GB)")
            
#         except subprocess.CalledProcessError:
#             raise ValidationError("Demucs not installed. Run: !pip install demucs")
    
#     def _validate_video(self, video_path: str):
#         """Validate video file exists"""
#         path = Path(video_path)
#         if not path.exists():
#             raise ValidationError(f"Video file not found: {video_path}")
#         if not path.is_file():
#             raise ValidationError(f"Not a file: {video_path}")
    
#     def extract_raw_audio(
#         self,
#         video_path: str,
#         output_path: str,
#         progress_callback: Optional[Callable[[str], None]] = None
#     ) -> str:
#         """
#         Stage 1: Extract raw audio from video (FFmpeg)
#         """
#         logger.info(f"[Stage 1] Extracting audio from: {Path(video_path).name}")
        
#         if progress_callback:
#             progress_callback("Extracting audio from video...")
        
#         # Always use mono for pipeline consistency (required by Pyannote later)
#         sample_rate = "22050" if self.fast_mode else "44100"
        
#         cmd = [
#             "ffmpeg", "-y", "-i", video_path,
#             "-vn",
#             "-acodec", "pcm_s16le",
#             "-ar", sample_rate,
#             "-ac", "1",  # Mono (required for diarization)
#             output_path
#         ]
        
#         try:
#             result = subprocess.run(
#                 cmd,
#                 capture_output=True,
#                 text=True,
#                 check=True,
#                 timeout=300
#             )
            
#             if not Path(output_path).exists():
#                 raise FFmpegError("FFmpeg completed but output file not found")
            
#             file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
#             logger.info(f"[Stage 1] ✓ Extracted: {file_size_mb:.1f}MB (mono, {sample_rate}Hz)")
            
#             return output_path
            
#         except subprocess.CalledProcessError as e:
#             raise FFmpegError(f"Audio extraction failed: {e.stderr}")
#         except subprocess.TimeoutExpired:
#             raise FFmpegError("Audio extraction timeout")
    
#     def separate_vocals(
#         self,
#         audio_path: str,
#         output_dir: str,
#         progress_callback: Optional[Callable[[str], None]] = None
#     ) -> Dict[str, str]:
#         """
#         Stage 2: AI-powered stem separation with Demucs (ALWAYS RUNS)
#         Uses soundfile backend to avoid torchcodec issues
#         """
#         logger.info("[Stage 2] Separating stems with Demucs...")
        
#         if progress_callback:
#             progress_callback("Separating vocals from background (AI running)...")
        
#         # Choose stable models
#         model = "htdemucs_ft" if not self.fast_mode else "htdemucs"
#         logger.info(f"Using model: {model}")
        
#         # CRITICAL: Force soundfile backend (fixes torchcodec crash)
#         env = dict(os.environ)
#         env["TORCHAUDIO_USE_BACKEND"] = "soundfile"
        
#         cmd = [
#             "demucs",
#             "--two-stems", "vocals",
#             "-n", model,
#             "--device", self.device,
#             "-o", output_dir,
#             audio_path
#         ]
        
#         if self.device == "cpu":
#             cmd.extend(["-j", "4"])  # Use 4 cores on CPU
        
#         try:
#             result = subprocess.run(
#                 cmd,
#                 capture_output=True,
#                 text=True,
#                 env=env,  # ← Use soundfile backend
#                 timeout=1800  # 30 min for long videos
#             )
            
#             if result.returncode != 0:
#                 logger.error("========== DEMUCS ERROR ==========")
#                 logger.error(result.stderr)
#                 raise DemucsError("Demucs failed. See logs above.")
            
#             # Auto-discover output folder (handles any filename)
#             model_dir = Path(output_dir) / model
#             if not model_dir.exists():
#                 raise DemucsError(f"Output directory not found: {model_dir}")
            
#             subfolders = [d for d in model_dir.iterdir() if d.is_dir()]
#             if not subfolders:
#                 raise DemucsError("No Demucs output folder found")
            
#             base_path = subfolders[0]
            
#             # Find vocal and background files (any extension)
#             vocals_files = list(base_path.glob("vocals.*"))
#             bg_files = list(base_path.glob("no_vocals.*"))
            
#             if not vocals_files:
#                 raise DemucsError(f"Vocals file not found in {base_path}")
#             if not bg_files:
#                 raise DemucsError(f"Background file not found in {base_path}")
            
#             vocals_path = str(vocals_files[0])
#             bg_path = str(bg_files[0])
            
#             logger.info(f"[Stage 2] ✓ Vocals: {Path(vocals_path).name}")
#             logger.info(f"[Stage 2] ✓ Background: {Path(bg_path).name}")
            
#             return {
#                 "vocals": vocals_path,
#                 "background": bg_path
#             }
            
#         except subprocess.TimeoutExpired:
#             raise DemucsError("Demucs timeout (video too long)")
    
#     def process_for_dubbing(
#         self,
#         video_path: str,
#         output_dir: Optional[str] = None,
#         progress_callback: Optional[Callable[[str], None]] = None,
#         cleanup_on_error: bool = True
#     ) -> Tuple[str, Optional[str], Dict]:
#         """
#         Complete processing pipeline for dubbing workflow
#         """
#         self._validate_video(video_path)
        
#         if output_dir is None:
#             output_dir = f"dubbing_temp_{self.job_id}"
        
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
        
#         logger.info(f"Processing job: {self.job_id}")
#         logger.info(f"Output directory: {output_dir}")
        
#         try:
#             # Stage 1: Extract audio
#             raw_audio = str(output_path / "raw_audio.wav")
#             self.extract_raw_audio(video_path, raw_audio, progress_callback)
            
#             # Stage 2: Separate stems (ALWAYS RUNS)
#             stems = self.separate_vocals(raw_audio, str(output_path), progress_callback)
            
#             metadata = {
#                 "job_id": self.job_id,
#                 "video_source": video_path,
#                 "quality_mode": "pro",  # ← Always pro
#                 "fast_mode": self.fast_mode,
#                 "has_background_track": True,  # ← Always has background
#                 "device_used": self.device,
#                 "output_directory": str(output_path),
#                 "vocals_file": stems["vocals"],
#                 "background_file": stems["background"]
#             }
            
#             if progress_callback:
#                 progress_callback("Processing complete!")
            
#             return stems["vocals"], stems["background"], metadata
            
#         except Exception as e:
#             logger.error(f"Processing failed: {e}")
            
#             if cleanup_on_error:
#                 logger.info("Cleaning up temporary files...")
#                 self.cleanup(output_dir)
            
#             raise
    
#     def remix_final_audio(
#         self,
#         dubbed_vocals: str,
#         original_background: str,  # ← No longer Optional (always exists)
#         output_path: str,
#         voice_volume: float = 1.0,
#         bg_volume: float = 0.7
#     ) -> str:
#         """
#         Stage 3: Mix dubbed voice with original background
#         """
#         logger.info("[Stage 3] Remixing final audio...")
        
#         if not Path(dubbed_vocals).exists():
#             raise ValidationError(f"Dubbed vocals not found: {dubbed_vocals}")
        
#         if not Path(original_background).exists():
#             raise ValidationError(f"Background track not found: {original_background}")
        
#         try:
#             # Professional remix (background always exists)
#             cmd = [
#                 "ffmpeg", "-y",
#                 "-i", dubbed_vocals,
#                 "-i", original_background,
#                 "-filter_complex",
#                 f"[0:a]volume={voice_volume}[voice];"
#                 f"[1:a]volume={bg_volume}[bg];"
#                 f"[voice][bg]amix=inputs=2:duration=first:dropout_transition=2",
#                 "-acodec", "aac", "-b:a", "192k",
#                 output_path
#             ]
            
#             subprocess.run(cmd, capture_output=True, check=True, timeout=300)
#             logger.info(f"[Stage 3] ✓ Final audio: {output_path}")
#             return output_path
            
#         except subprocess.CalledProcessError as e:
#             raise FFmpegError(f"Audio mixing failed: {e.stderr}")
#         except subprocess.TimeoutExpired:
#             raise FFmpegError("Audio mixing timeout")
    
#     def cleanup(self, temp_dir: str):
#         """Delete temporary files"""
#         path = Path(temp_dir)
#         if path.exists() and path.is_dir():
#             try:
#                 shutil.rmtree(path)
#                 logger.info(f"✓ Cleaned up: {temp_dir}")
#             except Exception as e:
#                 logger.warning(f"Cleanup failed: {e}")


# # ============================================================================
# # Colab-Specific Test Runner
# # ============================================================================

# def test_colab_pipeline():
#     """
#     Test pipeline for Google Colab (PRO MODE ONLY)
#     """
#     import sys
    
#     try:
#         # Initialize processor (PRO MODE mandatory)
#         processor = CinematicAudioProcessor(
#             fast_mode=True  # Only control speed, not quality mode
#         )
        
#         # Get video file (from upload or argument)
#         if len(sys.argv) > 1:
#             video_file = sys.argv[1]
#         else:
#             video_file = "video.mp4"
        
#         if not Path(video_file).exists():
#             print("\n⚠ Video file not found!")
#             print("In Colab, run this first:")
#             print("  from google.colab import files")
#             print("  uploaded = files.upload()")
#             print("  video_file = list(uploaded.keys())[0]")
#             return
        
#         # Progress callback
#         def progress(msg):
#             print(f"  → {msg}")
        
#         # Process video
#         print("\n" + "="*60)
#         print("Starting PRO dubbing pipeline...")
#         print("="*60 + "\n")
        
#         vocals, background, metadata = processor.process_for_dubbing(
#             video_file,
#             progress_callback=progress,
#             cleanup_on_error=True
#         )
        
#         print("\n" + "="*60)
#         print("✅ Processing Complete!")
#         print("="*60)
#         print(f"\nVocals: {vocals}")
#         print(f"Background: {background}")
#         print(f"\nMetadata:")
#         for key, value in metadata.items():
#             print(f"  {key}: {value}")
        
#         # Download results in Colab
#         try:
#             from google.colab import files
#             print("\n📥 Downloading results...")
#             files.download(vocals)
#             files.download(background)
#             print("✓ Files downloaded!")
#         except ImportError:
#             print("\n(Not in Colab - files saved locally)")
        
#         # Cleanup prompt
#         cleanup = input("\nDelete temporary files? (y/n): ")
#         if cleanup.lower() == 'y':
#             processor.cleanup(metadata["output_directory"])
        
#     except AudioProcessingError as e:
#         print(f"\n❌ Error: {e}")
#     except KeyboardInterrupt:
#         print("\n⚠ Interrupted by user")
#     except Exception as e:
#         print(f"\n❌ Unexpected error: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     test_colab_pipeline()


















# ============================================================================
# Trendy 2026 Audio Processing - Windows Compatible Version
# FIXED: Bypasses torchcodec by using Demucs as Python library + soundfile
# ============================================================================

import subprocess
import shutil
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable
import uuid
import os
import sys

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
# Demucs Python API (Bypasses CLI and torchcodec)
# ============================================================================
def run_demucs_separation(
    audio_path: str,
    output_dir: str,
    model_name: str = "htdemucs",
    device: str = "cpu",
    two_stems: str = "vocals"
) -> Dict[str, str]:
    """
    Run Demucs separation using Python API instead of CLI.
    Saves output using soundfile to bypass torchcodec issues.
    
    This is the CRITICAL fix for Windows compatibility.
    """
    import torch
    import soundfile as sf
    import numpy as np
    
    # Import Demucs components
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import AudioFile
    
    logger.info(f"Loading Demucs model: {model_name}")
    
    # Load model
    model = get_model(model_name)
    model.to(device)
    model.eval()
    
    # Get model sample rate
    samplerate = model.samplerate
    logger.info(f"Model sample rate: {samplerate}Hz")
    
    # ========================================================================
    # CRITICAL FIX: Use soundfile for LOADING (bypasses torchaudio/torchcodec)
    # ========================================================================
    logger.info(f"Loading audio: {audio_path}")
    
    # Load with soundfile (NOT torchaudio - avoids torchcodec completely)
    audio_data, sr = sf.read(audio_path, dtype='float32')
    
    # soundfile returns (samples, channels) or (samples,) for mono
    # Convert to torch tensor with shape (channels, samples)
    if audio_data.ndim == 1:
        # Mono: (samples,) -> (1, samples)
        wav = torch.from_numpy(audio_data).unsqueeze(0)
    else:
        # Stereo/Multi: (samples, channels) -> (channels, samples)
        wav = torch.from_numpy(audio_data.T)
    
    # Resample if needed using scipy (NOT torchaudio)
    if sr != samplerate:
        logger.info(f"Resampling from {sr}Hz to {samplerate}Hz")
        from scipy import signal
        num_samples = int(wav.shape[1] * samplerate / sr)
        resampled = []
        for ch in range(wav.shape[0]):
            resampled.append(signal.resample(wav[ch].numpy(), num_samples))
        wav = torch.from_numpy(np.stack(resampled))
    
    # Ensure stereo (Demucs expects stereo input)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # Mono to stereo
    elif wav.shape[0] > 2:
        wav = wav[:2]  # Take first 2 channels
    
    # Ensure float32 tensor
    wav = wav.float()
    
    # Add batch dimension: (channels, samples) -> (batch, channels, samples)
    wav = wav.unsqueeze(0).to(device)
    
    logger.info(f"Audio shape: {wav.shape}, Duration: {wav.shape[2]/samplerate:.1f}s")
    
    # Apply model with progress
    logger.info("Running AI separation (this may take several minutes)...")
    
    with torch.no_grad():
        # apply_model handles chunking internally for long audio
        sources = apply_model(
            model, 
            wav, 
            device=device,
            progress=True,  # Shows progress bar
            num_workers=0 if device == "cpu" else 4
        )
    
    # sources shape: (batch, num_sources, channels, samples)
    # For htdemucs with two_stems="vocals": sources are [drums, bass, other, vocals]
    # But with --two-stems vocals, we get [vocals, no_vocals]
    
    # Get source names from model
    source_names = model.sources
    logger.info(f"Model sources: {source_names}")
    
    # Find vocal index
    if "vocals" in source_names:
        vocal_idx = source_names.index("vocals")
    else:
        raise DemucsError(f"Model doesn't have vocals source. Sources: {source_names}")
    
    # Extract vocals and create "no_vocals" (sum of everything else)
    vocals = sources[0, vocal_idx]  # (channels, samples)
    
    # Sum all non-vocal sources for background
    non_vocal_indices = [i for i, name in enumerate(source_names) if name != "vocals"]
    background = sources[0, non_vocal_indices].sum(dim=0)  # (channels, samples)
    
    # Create output directory
    out_path = Path(output_dir) / model_name / Path(audio_path).stem
    out_path.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # CRITICAL: Save using soundfile (NOT torchaudio) to bypass torchcodec
    # ========================================================================
    vocals_file = out_path / "vocals.wav"
    background_file = out_path / "no_vocals.wav"
    
    logger.info(f"Saving vocals to: {vocals_file}")
    
    # Convert to numpy for soundfile
    # Shape: (channels, samples) -> (samples, channels) for soundfile
    vocals_np = vocals.cpu().numpy().T
    background_np = background.cpu().numpy().T
    
    # Normalize to prevent clipping
    vocals_np = np.clip(vocals_np, -1.0, 1.0)
    background_np = np.clip(background_np, -1.0, 1.0)
    
    # Save with soundfile (100% reliable, no torchcodec dependency)
    sf.write(str(vocals_file), vocals_np, samplerate, subtype='PCM_16')
    logger.info(f"✓ Saved vocals: {vocals_file}")
    
    sf.write(str(background_file), background_np, samplerate, subtype='PCM_16')
    logger.info(f"✓ Saved background: {background_file}")
    
    return {
        "vocals": str(vocals_file),
        "background": str(background_file)
    }


# ============================================================================
# Main Audio Processor
# ============================================================================
class CinematicAudioProcessor:
    """
    Professional-grade audio processing for movie dubbing.
    Windows-compatible version using soundfile for all audio I/O.
    
    Features:
    - Windows compatible (no torchcodec dependency)
    - GPU acceleration when available
    - Robust error handling
    - Progress callbacks
    - PRO MODE ONLY: Demucs stem separation always enabled
    """
    
    def __init__(self, fast_mode: bool = False):
        """
        Args:
            fast_mode: If True, use faster model (htdemucs instead of htdemucs_ft)
        
        Note: Pro mode (Demucs stem separation) is ALWAYS enabled
        """
        import torch
        
        self.quality_mode = "pro"  # ← MANDATORY PRO MODE
        self.fast_mode = fast_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.job_id = str(uuid.uuid4())[:8]
        
        # Choose model based on speed preference
        self.model_name = "htdemucs" if fast_mode else "htdemucs_ft"
        
        logger.info(f"AudioProcessor initialized:")
        logger.info(f"  Mode: PRO (mandatory)")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Fast mode: {fast_mode}")
        
        # Check dependencies
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Verify all required dependencies are installed"""
        # Check FFmpeg
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                timeout=5, 
                check=True
            )
            logger.info("✓ FFmpeg available")
        except Exception:
            raise ValidationError("FFmpeg not installed. Install with: choco install ffmpeg")
        
        # Check soundfile (CRITICAL for Windows compatibility)
        try:
            import soundfile as sf
            logger.info("✓ soundfile available")
        except ImportError:
            raise ValidationError("soundfile not installed. Run: pip install soundfile")
        
        # Check Demucs
        try:
            from demucs.pretrained import get_model
            logger.info("✓ Demucs available")
        except ImportError:
            raise ValidationError("Demucs not installed. Run: pip install demucs")
        
        # Check torch
        try:
            import torch
            logger.info(f"✓ PyTorch available (version {torch.__version__})")
        except ImportError:
            raise ValidationError("PyTorch not installed")
        
        logger.info("Note: First run will download AI model (~2GB)")
    
    def _validate_video(self, video_path: str):
        """Validate video file exists"""
        path = Path(video_path)
        if not path.exists():
            raise ValidationError(f"Video file not found: {video_path}")
        if not path.is_file():
            raise ValidationError(f"Not a file: {video_path}")
        
        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Input video: {path.name} ({size_mb:.1f}MB)")
    
    def extract_raw_audio(
        self,
        video_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Stage 1: Extract raw audio from video using FFmpeg
        
        Output: WAV file (mono, 22050Hz for fast mode, 44100Hz otherwise)
        """
        logger.info(f"[Stage 1] Extracting audio from: {Path(video_path).name}")
        
        if progress_callback:
            progress_callback("Extracting audio from video...")
        
        # Sample rate based on mode
        sample_rate = "22050" if self.fast_mode else "44100"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",                    # No video
            "-acodec", "pcm_s16le",   # 16-bit PCM
            "-ar", sample_rate,       # Sample rate
            "-ac", "1",               # Mono (required for diarization later)
            output_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 min timeout
            )
            
            if not Path(output_path).exists():
                raise FFmpegError("FFmpeg completed but output file not found")
            
            file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"[Stage 1] ✓ Extracted: {file_size_mb:.1f}MB (mono, {sample_rate}Hz)")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise FFmpegError(f"Audio extraction failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise FFmpegError("Audio extraction timeout (video too long?)")
    
    def separate_vocals(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, str]:
        """
        Stage 2: AI-powered stem separation with Demucs
        
        Uses Python API + soundfile to completely bypass torchcodec issues.
        This is the key fix for Windows compatibility.
        """
        logger.info("[Stage 2] Separating stems with Demucs...")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Device: {self.device}")
        
        if progress_callback:
            progress_callback(f"Separating vocals from background ({self.model_name})...")
        
        try:
            # Use our custom function that bypasses torchcodec
            stems = run_demucs_separation(
                audio_path=audio_path,
                output_dir=output_dir,
                model_name=self.model_name,
                device=self.device,
                two_stems="vocals"
            )
            
            logger.info(f"[Stage 2] ✓ Vocals: {Path(stems['vocals']).name}")
            logger.info(f"[Stage 2] ✓ Background: {Path(stems['background']).name}")
            
            return stems
            
        except Exception as e:
            logger.error(f"Demucs separation failed: {e}")
            import traceback
            traceback.print_exc()
            raise DemucsError(f"Stem separation failed: {e}")
    
    def process_for_dubbing(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        cleanup_on_error: bool = True
    ) -> Tuple[str, str, Dict]:
        """
        Complete processing pipeline for dubbing workflow.
        
        Returns:
            Tuple of (vocals_path, background_path, metadata_dict)
        """
        self._validate_video(video_path)
        
        # Create output directory
        if output_dir is None:
            output_dir = f"dubbing_output_{self.job_id}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing job: {self.job_id}")
        logger.info(f"Output directory: {output_dir}")
        
        try:
            # Stage 1: Extract audio from video
            raw_audio = str(output_path / "raw_audio.wav")
            self.extract_raw_audio(video_path, raw_audio, progress_callback)
            
            # Stage 2: Separate vocals and background
            stems = self.separate_vocals(raw_audio, str(output_path), progress_callback)
            
            # Build metadata
            metadata = {
                "job_id": self.job_id,
                "video_source": str(Path(video_path).absolute()),
                "quality_mode": "pro",
                "model_used": self.model_name,
                "fast_mode": self.fast_mode,
                "device_used": self.device,
                "output_directory": str(output_path.absolute()),
                "raw_audio": raw_audio,
                "vocals_file": stems["vocals"],
                "background_file": stems["background"]
            }
            
            if progress_callback:
                progress_callback("✅ Processing complete!")
            
            logger.info("[Pipeline] ✓ All stages completed successfully!")
            
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
        original_background: str,
        output_path: str,
        voice_volume: float = 1.0,
        bg_volume: float = 0.7
    ) -> str:
        """
        Stage 3: Mix dubbed voice with original background music
        
        Args:
            dubbed_vocals: Path to the dubbed/translated voice track
            original_background: Path to the original background music
            output_path: Where to save the final mixed audio
            voice_volume: Volume multiplier for voice (default 1.0)
            bg_volume: Volume multiplier for background (default 0.7)
        """
        logger.info("[Stage 3] Remixing final audio...")
        
        # Validate inputs
        if not Path(dubbed_vocals).exists():
            raise ValidationError(f"Dubbed vocals not found: {dubbed_vocals}")
        
        if not Path(original_background).exists():
            raise ValidationError(f"Background track not found: {original_background}")
        
        try:
            # Use FFmpeg for professional audio mixing
            filter_complex = (
                f"[0:a]volume={voice_volume}[voice];"
                f"[1:a]volume={bg_volume}[bg];"
                f"[voice][bg]amix=inputs=2:duration=first:dropout_transition=2"
            )
            
            cmd = [
                "ffmpeg", "-y",
                "-i", dubbed_vocals,
                "-i", original_background,
                "-filter_complex", filter_complex,
                "-acodec", "aac",
                "-b:a", "192k",
                output_path
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                check=True, 
                timeout=300
            )
            
            file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"[Stage 3] ✓ Final audio: {output_path} ({file_size_mb:.1f}MB)")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise FFmpegError(f"Audio mixing failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise FFmpegError("Audio mixing timeout")
    
    def cleanup(self, temp_dir: str):
        """Delete temporary processing files"""
        path = Path(temp_dir)
        if path.exists() and path.is_dir():
            try:
                shutil.rmtree(path)
                logger.info(f"✓ Cleaned up: {temp_dir}")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")


# ============================================================================
# Test Runner
# ============================================================================
def main():
    """
    Test the audio processing pipeline.
    """
    print("\n" + "="*70)
    print("🎬 Cinematic Audio Processor - Windows Compatible Version")
    print("="*70 + "\n")
    
    try:
        # Initialize processor
        processor = CinematicAudioProcessor(
            fast_mode=True  # Use htdemucs for faster processing
        )
        
        # Get video file
        if len(sys.argv) > 1:
            video_file = sys.argv[1]
        else:
            video_file = "video.mp4"
        
        if not Path(video_file).exists():
            print(f"\n⚠ Video file not found: {video_file}")
            print("\nUsage:")
            print("  python core.py <video_file>")
            print("  python core.py video.mp4")
            return
        
        # Progress callback
        def progress(msg):
            print(f"  → {msg}")
        
        # Process video
        print("\n" + "-"*70)
        print("Starting PRO dubbing pipeline...")
        print("-"*70 + "\n")
        
        vocals, background, metadata = processor.process_for_dubbing(
            video_file,
            progress_callback=progress,
            cleanup_on_error=True
        )
        
        # Success!
        print("\n" + "="*70)
        print("✅ PROCESSING COMPLETE!")
        print("="*70)
        print(f"\n📁 Output Files:")
        print(f"   Vocals:     {vocals}")
        print(f"   Background: {background}")
        print(f"\n📊 Metadata:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
        
        # Cleanup prompt
        print("\n" + "-"*70)
        cleanup = input("Delete temporary files? (y/n): ").strip().lower()
        if cleanup == 'y':
            processor.cleanup(metadata["output_directory"])
            print("✓ Cleanup complete!")
        else:
            print(f"Files kept in: {metadata['output_directory']}")
        
    except AudioProcessingError as e:
        print(f"\n❌ Processing Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()