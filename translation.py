
import os
import json
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass
from datetime import datetime
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class TranslationConfig:
    """Configuration for translation engine"""
    groq_api_key: str = ""
    model: str = "meta-llama/llama-3.3-70b-instruct"
    batch_size: int = None
    max_syllable_rate: float = 4.0
    source_language: str = "tr"  # ADD THIS
    target_language: str = "en"  # ADD THIS
    retry_attempts: int = 3
    rate_limit_delay: float = 0.5
    
    # Token limits for mega-batch calculation
    max_input_tokens: int = 120000  # Safe limit
    max_output_tokens: int = 7000   # Groq limit with buffer
    avg_turkish_tokens_per_segment: int = 25
    avg_english_tokens_per_segment: int = 20
    prompt_overhead_tokens: int = 1000
    
    def __post_init__(self):
        if not self.groq_api_key:
            self.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        
        # Auto-calculate optimal batch size if not set
        if self.batch_size is None:
            self.batch_size = self._calculate_optimal_batch_size()
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate maximum segments per batch based on token limits"""
        # Input constraint
        available_input = self.max_input_tokens - self.prompt_overhead_tokens
        max_segments_input = available_input // self.avg_turkish_tokens_per_segment
        
        # Output constraint
        max_segments_output = self.max_output_tokens // self.avg_english_tokens_per_segment
        
        # Take the smaller limit and add 20% safety margin
        safe_batch_size = int(min(max_segments_input, max_segments_output) * 0.8)
        
        # Reasonable bounds
        return max(50, min(safe_batch_size, 350))


# ============================================================================
# Syllable Counter (English)
# ============================================================================
class SyllableCounter:
    """
    Estimates English syllable count using linguistic heuristics.
    Accuracy: ~90% (sufficient for dubbing timing)
    """
    
    # Common word overrides (irregular syllable counts)
    OVERRIDES = {
        "every": 2, "different": 3, "interesting": 4, "comfortable": 3,
        "favorite": 3, "actually": 4, "probably": 3, "family": 3,
        "beautiful": 3, "business": 2, "chocolate": 3, "camera": 3,
        "evening": 2, "everything": 3, "heaven": 2, "listen": 2,
        "natural": 3, "prisoner": 3, "restaurant": 3, "several": 3,
        "temperature": 4, "vegetable": 4, "wednesday": 2, "didn't": 2,
        "wouldn't": 2, "couldn't": 2, "shouldn't": 2, "haven't": 2,
        "wasn't": 2, "weren't": 2, "isn't": 2, "aren't": 2,
        "don't": 1, "won't": 1, "can't": 1, "i'm": 1, "you're": 1,
        "we're": 1, "they're": 1, "what's": 1, "that's": 1, "it's": 1,
        "there's": 1, "here's": 1, "let's": 1, "gonna": 2, "gotta": 2,
        "wanna": 2, "kinda": 2, "sorta": 2, "coulda": 2, "woulda": 2,
        "shoulda": 2, "oughta": 2, "gimme": 2, "lemme": 2, "dunno": 2,
    }
    
    @classmethod
    def count(cls, text: str) -> int:
        """Count syllables in English text"""
        if not text or not text.strip():
            return 0
        
        text = text.lower().strip()
        words = re.findall(r"[a-z']+", text)
        
        total = 0
        for word in words:
            if word in cls.OVERRIDES:
                total += cls.OVERRIDES[word]
                continue
            
            word_clean = re.sub(r"[^a-z]", "", word)
            if not word_clean:
                continue
            
            vowel_groups = re.findall(r"[aeiouy]+", word_clean)
            count = len(vowel_groups)
            
            if word_clean.endswith('e') and len(word_clean) > 2 and count > 1:
                count -= 1
            
            if word_clean.endswith('le') and len(word_clean) > 2:
                if word_clean[-3] not in 'aeiouy':
                    count += 1
            
            total += max(1, count)
        
        return total
    
    @classmethod
    def validate_timing(cls, text: str, duration: float, max_rate: float = 4.0) -> Dict:
        """Check if text fits within duration at given syllable rate"""
        syllables = cls.count(text)
        rate = syllables / duration if duration > 0 else float('inf')
        
        return {
            "syllables": syllables,
            "duration": round(duration, 2),
            "rate": round(rate, 2),
            "budget": int(duration * max_rate),
            "valid": rate <= max_rate,
            "overflow": max(0, syllables - int(duration * max_rate))
        }


# ============================================================================
# Mega-Batch Manager (Smart Chunking)
# ============================================================================
class MegaBatchManager:
    """
    Creates large batches (50-350 segments) instead of small ones (10).
    Reduces API calls from 150+ to 5-10 while respecting token limits.
    """
    
    def __init__(self, batch_size: int = 250):
        self.batch_size = batch_size
        logger.info(f" Mega-Batch Mode: {batch_size} segments/batch")
    
    def create_batches(self, segments: List[Dict]) -> Generator[tuple, None, None]:
        """
        Yield (batch_index, batch) tuples.
        Returns batch index for checkpoint tracking.
        """
        total_segments = len(segments)
        batch_idx = 0
        
        for i in range(0, total_segments, self.batch_size):
            batch = []
            for seg in segments[i:i + self.batch_size]:
                seg_with_duration = dict(seg)
                seg_with_duration['duration'] = round(
                    seg.get('end', 0) - seg.get('start', 0), 2
                )
                batch.append(seg_with_duration)
            
            yield (batch_idx, batch)
            batch_idx += 1


# ============================================================================
# Groq Translator (LLM Interface)
# ============================================================================
class GroqTranslator:
    """
    Interfaces with Groq API for translation.
    Uses the Syllable-Budget Master Prompt v2.0
    """
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.source_lang = config.source_language  # ADD THIS
        self.target_lang = config.target_language  # ADD THIS
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenRouter client (OpenAI-compatible)"""
        if not self.config.groq_api_key:
            logger.warning("OPENROUTER_API_KEY not set!")
            return
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.config.groq_api_key  # Using same config field for OpenRouter key
            )
            logger.info("✓ OpenRouter client initialized (FREE tier)")
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter: {e}")
    
    def _build_prompt(self, batch: List[Dict]) -> str:
        """Build the Syllable-Budget Master Prompt v2.0"""
        
        # Minimal batch data for token efficiency
        batch_data = [
            {
                "idx": i,
                "speaker": seg.get("speaker", "SPEAKER_00"),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "duration": seg.get("duration", 0),
                "budget": int(seg.get("duration", 0) * 4),
                "text": seg.get("text", "")
            }
            for i, seg in enumerate(batch)
        ]
        
        prompt = f"""### ROLE
You are a Lead Script Adapter for a Hollywood Dubbing Studio.
Your specialty is {self.source_lang}-to-{self.target_lang} localization for lip-sync dubbing.

### CORE RULES
- 1 Second = 4.0 Syllables MAX (Standard Speech Rate)
- Prioritize 'Acting Beat' over 'Literal Translation'
- English is 20% denser than Turkish—AGGRESSIVELY shorten phrases

### ULTRA-SHORT SEGMENTS (<0.5 sec)
- Use single reactions: "What?" "No!" "Huh?" "Hey!" "Stop!"
- Capture REACTION SOUND, not literal meaning

### CHARACTER VOICE CONSISTENCY
- Formal Turkish ("Efendim") → Formal English ("Sir", "Pardon")
- Street Turkish ("Lan", "Abi") → Casual English ("Man", "Dude")
- SAME speaker = SAME voice style across all segments

### CULTURAL RULES
- Preserve proper names from source language (do NOT anglicize)
- Adapt culture-specific phrases to target language equivalents
- Maintain emotional intensity and character voice consistency

### TURKISH DRAMA EMOTIONS
- Öfke (Rage): Short, punchy. "Get out. NOW."
- Hüzün (Sorrow): Soft, trailing. "I just... can't."
- Şok (Shock): Fragmented. "You... WHAT?"
- Aşk (Love): Breathless. "I'd die for you."

### INPUT BATCH ({len(batch)} SEGMENTS)
{json.dumps(batch_data, ensure_ascii=False, indent=2)}

### EMOTION ADAPTATION
- Rage: Short, punchy. "Get out. NOW."
- Sorrow: Soft, trailing. "I just... can't."
- Shock: Fragmented. "You... WHAT?"
- Love: Breathless. "I'd die for you."

### EXECUTION PER SEGMENT
1. CALCULATE: duration × 4 = syllable budget
2. ANALYZE: Core emotion + character voice
3. DRAFT: Create translation
4. AUDIT: Count syllables. If over budget → use contractions
5. VERIFY: Must fit within duration

### CONTRACTION SHORTCUTS
- "Have to" → "Gotta" | "Want to" → "Wanna"
- "Be careful" → "Watch it" | "I cannot" → "I can't"
- "What are you" → "What're you" | "Do not" → "Don't"

### OUTPUT FORMAT (STRICT JSON - MUST RETURN ALL {len(batch)} SEGMENTS)
Return a JSON object with "segments" array containing exactly {len(batch)} items:

{{
  "segments": [
    {{
      "idx": 0,
      "speaker": "SPEAKER_XX",
      "start": X.XX,
      "end": X.XX,
      "duration": X.XX,
      "budget": N,
      "original": "Turkish text",
      "translated": "English text",
      "syllables": N,
      "emotion": "anger/fear/love/shock/sorrow/neutral"
    }},
    ... (continue for all {len(batch)} segments)
  ]
}}

CRITICAL: The "segments" array MUST contain exactly {len(batch)} items, one for each input segment.
"""
        
        return prompt
    
    def translate_batch(self, batch: List[Dict]) -> List[Dict]:
        """Translate a batch of segments"""
        if not self.client:
            logger.error("OpenRouter client not initialized")
            return self._fallback_translate(batch)
        
        prompt = self._build_prompt(batch)
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.config.model,
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=32000,  # Increased for mega-batches
                    extra_headers={
                        "HTTP-Referer": "https://github.com/multimodal-video-dubbing",
                        "X-Title": "Video Dubbing Assistant"
                    }
                )
                
                content = response.choices[0].message.content
                result = json.loads(content)
                
                # Debug: Log the raw response structure
                logger.debug(f"Raw response keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
                
                # Handle wrapped responses - check ALL possible array keys
                if isinstance(result, dict):
                    # Try all possible wrapper keys
                    for key in ["segments", "translations", "data", "results", "output", "items"]:
                        if key in result and isinstance(result[key], list):
                            result = result[key]
                            logger.info(f"   Extracted array from '{key}' wrapper")
                            break
                    else:
                        # If no array found, check if dict has numeric keys (0, 1, 2...)
                        numeric_keys = [k for k in result.keys() if k.isdigit()]
                        if numeric_keys:
                            result = [result[str(i)] for i in range(len(numeric_keys))]
                            logger.info(f"   Reconstructed array from {len(numeric_keys)} numeric keys")
                        else:
                            # Last resort: wrap single dict
                            logger.warning(f"   Single dict response with keys: {list(result.keys())}")
                            result = [result]
                
                if isinstance(result, list) and len(result) == len(batch):
                    # Sort by idx to ensure order
                    result_sorted = sorted(result, key=lambda x: x.get('idx', 0))
                    return result_sorted
                else:
                    logger.warning(f"Batch size mismatch: expected {len(batch)}, got {len(result) if isinstance(result, list) else 'non-list'}")
                    
                    # Save problematic response for debugging
                    debug_file = f"debug_response_batch{attempt}.json"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        json.dump({"expected": len(batch), "got": result, "raw_content": content}, f, indent=2, ensure_ascii=False)
                    logger.warning(f"   Saved debug response to: {debug_file}")
                    
                    if isinstance(result, list) and len(result) > 0:
                        return self._merge_results(batch, result)
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.warning(f"API error (attempt {attempt + 1}): {e}")
            
            if attempt < self.config.retry_attempts - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        
        logger.error(f"All retries failed for batch of {len(batch)} segments")
        return self._fallback_translate(batch)
    
    def _merge_results(self, batch: List[Dict], results: List[Dict]) -> List[Dict]:
        """Merge partial results with original batch"""
        merged = []
        for i, seg in enumerate(batch):
            if i < len(results):
                merged.append(results[i])
            else:
                merged.append(self._fallback_segment(seg))
        return merged
    
    def _fallback_translate(self, batch: List[Dict]) -> List[Dict]:
        """Fallback when API fails"""
        return [self._fallback_segment(seg) for seg in batch]
    
    def _fallback_segment(self, seg: Dict) -> Dict:
        """Create fallback translation for single segment"""
        duration = seg.get('duration', seg.get('end', 0) - seg.get('start', 0))
        return {
            "speaker": seg.get("speaker", "SPEAKER_00"),
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "duration": round(duration, 2),
            "budget": int(duration * 4),
            "original": seg.get("text", ""),
            "translated": f"[NEEDS_REVIEW] {seg.get('text', '')}",
            "syllables": 0,
            "emotion": "unknown",
            "error": "translation_failed"
        }


# ============================================================================
# Translation Engine (Mega-Batch + Bulletproof Checkpoint)
# ============================================================================
class TranslationEngine:
    """
    Production-Grade Translation Pipeline - MEGA-BATCH OPTIMIZED
    
    Features:
    - Mega-batch processing (50-350 segments per API call)
    - Reduces 150+ calls → 5-10 calls (10-15× faster)
    - Bulletproof checkpoint after EVERY batch
    - Atomic file writes for crash safety
    """
    
    def __init__(self, config: Optional[TranslationConfig] = None):
        self.config = config or TranslationConfig()
        self.batch_manager = MegaBatchManager(batch_size=self.config.batch_size)
        self.translator = GroqTranslator(self.config)
        self.syllable_counter = SyllableCounter()
    
    def load_transcript(self, path: str) -> Dict:
        """Load transcript JSON"""
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
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, path: str) -> Optional[Dict]:
        """Load checkpoint if exists"""
        checkpoint_path = path + ".checkpoint"
        if Path(checkpoint_path).exists():
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"    Checkpoint found:")
                    logger.info(f"      Last batch: {data.get('last_batch', 0)}")
                    logger.info(f"      Segments done: {len(data.get('segments', []))}")
                    logger.info(f"      Time: {data.get('saved_at', 'unknown')}")
                    return data
            except Exception as e:
                logger.warning(f"     Corrupted checkpoint: {e}")
                return None
        return None
    
    def validate_and_fix(self, segment: Dict) -> Dict:
        """Validate timing and add metadata"""
        translated = segment.get("translated", "")
        duration = segment.get("duration", 0)
        
        timing = self.syllable_counter.validate_timing(
            translated, duration, self.config.max_syllable_rate
        )
        
        segment["syllables"] = timing["syllables"]
        segment["syllable_rate"] = timing["rate"]
        segment["timing_valid"] = timing["valid"]
        
        if not timing["valid"]:
            segment["overflow"] = timing["overflow"]
            segment["needs_shortening"] = True
        
        return segment
    
    def translate(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        resume: bool = True
    ) -> Dict:
        """
        Main translation pipeline - MEGA-BATCH MODE
        """
        
        logger.info("=" * 60)
        logger.info(" Translation Engine - Stage 4 (MEGA-BATCH)")
        logger.info(f"   Model: {self.config.model}")
        logger.info(f"   Batch size: {self.config.batch_size} segments")
        logger.info(f"   Checkpoint: EVERY batch (bulletproof)")
        logger.info("=" * 60)
        
        # Load transcript
        logger.info(f"\n[1/4] Loading transcript: {input_path}")
        transcript = self.load_transcript(input_path)
        segments = transcript.get("segments", [])
        
        if not segments:
            logger.error("No segments found!")
            return {"error": "no_segments"}
        
        logger.info(f"   Found {len(segments)} segments")
        
        # Output path
        if output_path is None:
            output_path = str(Path(input_path).parent / "translated.json")
        
        # Load checkpoint
        translated_segments = []
        start_batch_idx = 0
        
        if resume:
            checkpoint = self.load_checkpoint(output_path)
            if checkpoint:
                translated_segments = checkpoint.get("segments", [])
                start_batch_idx = checkpoint.get("last_batch", 0) + 1
                logger.info(f"    Resuming from batch {start_batch_idx}")
        
        # Create batches
        logger.info(f"\n[2/4] Creating mega-batches...")
        batch_generator = self.batch_manager.create_batches(segments)
        
        # Count total batches
        total_batches = (len(segments) + self.config.batch_size - 1) // self.config.batch_size
        logger.info(f"   {total_batches} mega-batches (was {len(segments)//10} small batches)")
        logger.info(f"    {(len(segments)//10) / total_batches:.1f}× fewer API calls!")
        
        if start_batch_idx > 0:
            logger.info(f"    Skipping {start_batch_idx} completed batches")
        
        # Stats
        stats = {
            "total_segments": len(segments),
            "processed_segments": len(translated_segments),
            "timing_valid": sum(1 for s in translated_segments if s.get("timing_valid", True)),
            "timing_invalid": sum(1 for s in translated_segments if not s.get("timing_valid", True)),
            "errors": sum(1 for s in translated_segments if s.get("error")),
            "retries": 0
        }
        
        # Process batches
        logger.info(f"\n[3/4] Translating...")
        
        try:
            for batch_idx, batch in batch_generator:
                # Skip already processed batches
                if batch_idx < start_batch_idx:
                    continue
                
                progress = (batch_idx + 1) / total_batches * 100
                segments_range = f"{len(translated_segments)}-{len(translated_segments) + len(batch)}"
                logger.info(f"   Batch {batch_idx + 1}/{total_batches} ({progress:.1f}%) - Segments {segments_range}")
                
                # Translate
                try:
                    results = self.translator.translate_batch(batch)
                    
                    # Validate
                    for seg in results:
                        validated = self.validate_and_fix(seg)
                        translated_segments.append(validated)
                        
                        if validated.get("timing_valid", True):
                            stats["timing_valid"] += 1
                        else:
                            stats["timing_invalid"] += 1
                        
                        if validated.get("error"):
                            stats["errors"] += 1
                    
                    stats["processed_segments"] = len(translated_segments)
                    
                except Exception as e:
                    logger.error(f"    Batch {batch_idx + 1} failed: {e}")
                    stats["errors"] += len(batch)
                    # Fallback
                    fallback_results = self.translator._fallback_translate(batch)
                    for seg in fallback_results:
                        validated = self.validate_and_fix(seg)
                        translated_segments.append(validated)
                
                #  CHECKPOINT AFTER EVERY BATCH 
                checkpoint_data = {
                    "last_batch": batch_idx,
                    "segments": translated_segments,
                    "saved_at": datetime.now().isoformat(),
                    "stats": stats
                }
                self.save_checkpoint(output_path, checkpoint_data)
                logger.info(f"    Checkpoint: {len(translated_segments)}/{len(segments)} segments saved")
                
                # Rate limit
                time.sleep(self.config.rate_limit_delay)
        
        except KeyboardInterrupt:
            logger.warning("\n  Interrupted!")
            logger.info(f"    Saved: {len(translated_segments)} segments")
            logger.info(f"   Resume: python translation.py {input_path}")
            
            checkpoint_data = {
                "last_batch": batch_idx,
                "segments": translated_segments,
                "saved_at": datetime.now().isoformat(),
                "interrupted": True,
                "stats": stats
            }
            self.save_checkpoint(output_path, checkpoint_data)
            
            return {
                "interrupted": True,
                "segments_saved": len(translated_segments),
                "last_batch": batch_idx
            }
        
        # Final save
        logger.info(f"\n[4/4] Saving final results...")
        
        result = {
            "source_language": transcript.get("language", "tr"),
            "target_language": "en",
            "num_speakers": transcript.get("num_speakers", 2),
            "total_duration": transcript.get("total_duration", 0),
            "translation_model": self.config.model,
            "timestamp": datetime.now().isoformat(),
            "segments": translated_segments,
            "stats": stats
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"   ✓ Saved: {output_path}")
        
        # Cleanup
        checkpoint_path = output_path + ".checkpoint"
        if Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()
            logger.info(f"     Checkpoint cleaned")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info(" Translation Complete!")
        logger.info(f"   Segments: {stats['total_segments']}")
        logger.info(f"   Timing Valid: {stats['timing_valid']} ({stats['timing_valid']/stats['total_segments']*100:.1f}%)")
        logger.info(f"   Timing Invalid: {stats['timing_invalid']}")
        logger.info(f"   Errors: {stats['errors']}")
        logger.info(f"   API Calls: {total_batches} (saved {(len(segments)//10) - total_batches} calls)")
        logger.info("=" * 60)
        
        return result


# ============================================================================
# CLI Entry Point
# ============================================================================
def main():
    import sys
    
    input_path = sys.argv[1] if len(sys.argv) > 1 else "transcript.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    source_lang = sys.argv[3] if len(sys.argv) > 3 else "auto"  
    target_lang = sys.argv[4] if len(sys.argv) > 4 else "telugu"    
    
    if not Path(input_path).exists():
        print(f" Not found: {input_path}")
        print("\nUsage: python translation.py <transcript.json> [output.json] [source_lang] [target_lang]")
        print("Example: python translation.py transcript.json translated.json tr en")
        print("         python translation.py transcript.json translated.json es fr")
        return
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(" OPENROUTER_API_KEY not set!")
        print("\nSet it:")
        print('  Windows: $env:OPENROUTER_API_KEY="sk-or-v1-..."')
        print('  Linux/Mac: export OPENROUTER_API_KEY="sk-or-v1-..."')
        print("\nGet your FREE API key at: https://openrouter.ai/")
        return
    
    print(f"✓ OPENROUTER_API_KEY found")
    print(f"✓ Input: {input_path}")
    
    try:
        config = TranslationConfig(
            groq_api_key=api_key,
            source_language=source_lang,  
            target_language=target_lang 
        )
        print(f"✓ Auto-calculated batch size: {config.batch_size} segments")
        
        engine = TranslationEngine(config)
        result = engine.translate(input_path, output_path)
        
        if result.get("segments"):
            print("\n Sample Output:")
            print("-" * 50)
            for seg in result["segments"][:5]:
                print(f"[{seg.get('start', 0):7.2f}s] {seg.get('speaker', '?')}:")
                print(f"   TR: {seg.get('original', 'N/A')}")
                print(f"   EN: {seg.get('translated', 'N/A')}")
                print(f"     {seg.get('syllables', 0)} syl / {seg.get('duration', 0):.2f}s = {seg.get('syllable_rate', 0):.1f} syl/s")
                print()
            
            if len(result["segments"]) > 5:
                print(f"   ... and {len(result['segments']) - 5} more segments")
    
    except KeyboardInterrupt:
        print("\n\n Interrupted! Progress saved.")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()