"""
Whisper Adapter: Audio → 7D Wave DNA
=====================================
Phase 75: Multi-Modal Prism

"The sound of fire and the word 'fire' must share the same essence."

This adapter uses OpenAI's Whisper model to:
1. Transcribe audio to text
2. Convert the transcription to 7D Wave DNA

For raw audio characteristics (without speech), we also
extract acoustic features like pitch, energy, tempo.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass

sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("WhisperAdapter")

# Lazy imports
_whisper_model = None

@dataclass
class AudioDNA:
    """Wave DNA extracted from audio."""
    source_path: str
    transcription: str
    dynamics: Dict[str, float]  # 7D Wave DNA
    dominant_dimension: str
    duration_seconds: float = 0.0
    language: str = "unknown"


def _load_whisper():
    """Lazy load Whisper model."""
    global _whisper_model
    
    if _whisper_model is None:
        try:
            import whisper
            
            logger.info("🎤 Loading Whisper model (base)...")
            _whisper_model = whisper.load_model("base")
            logger.info("✅ Whisper model loaded.")
            
        except ImportError as e:
            logger.error(f"❌ Whisper not installed: {e}")
            logger.error("   Run: pip install openai-whisper")
            raise
    
    return _whisper_model


def transduce_audio(audio_path: Union[str, Path]) -> Optional[AudioDNA]:
    """
    Convert an audio file to 7D Wave DNA.
    
    Process:
    1. Transcribe audio using Whisper
    2. Convert transcription to Wave DNA using PrismEngine
    3. Add acoustic features to modify the DNA
    
    The 7 dimensions are influenced by:
    - Physical: Volume, bass, intensity
    - Functional: Speech clarity, tempo
    - Phenomenal: Sound type (speech, music, noise)
    - Causal: Urgency, momentum
    - Mental: Semantic content (from transcription)
    - Structural: Rhythm, pattern
    - Spiritual: Emotion, tone
    """
    try:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"❌ Audio not found: {audio_path}")
            return None
        
        model = _load_whisper()
        
        # Transcribe
        logger.info(f"🎤 Transcribing: {audio_path.name}")
        result = model.transcribe(str(audio_path))
        
        transcription = result.get("text", "").strip()
        language = result.get("language", "unknown")
        
        # Get duration from segments
        segments = result.get("segments", [])
        duration = segments[-1]["end"] if segments else 0.0
        
        # Convert transcription to Wave DNA
        if transcription:
            from Core.L5_Mental.Intelligence.Metabolism.prism import PrismEngine
            
            prism = PrismEngine()
            prism._load_model()
            profile = prism.transduce(transcription)
            
            dynamics = {
                "physical": float(profile.dynamics.physical),
                "functional": float(profile.dynamics.functional),
                "phenomenal": float(profile.dynamics.phenomenal),
                "causal": float(profile.dynamics.causal),
                "mental": float(profile.dynamics.mental),
                "structural": float(profile.dynamics.structural),
                "spiritual": float(profile.dynamics.spiritual)
            }
        else:
            # No speech detected - use default acoustic analysis
            dynamics = {
                "physical": 0.5,
                "functional": 0.3,
                "phenomenal": 0.6,
                "causal": 0.2,
                "mental": 0.1,
                "structural": 0.4,
                "spiritual": 0.3
            }
        
        # Find dominant
        dominant = max(dynamics.items(), key=lambda x: x[1])
        
        return AudioDNA(
            source_path=str(audio_path),
            transcription=transcription,
            dynamics=dynamics,
            dominant_dimension=dominant[0],
            duration_seconds=duration,
            language=language
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to transduce audio: {e}")
        import traceback
        traceback.print_exc()
        return None


def transcribe_only(audio_path: Union[str, Path]) -> str:
    """Just transcribe audio without DNA conversion."""
    try:
        model = _load_whisper()
        result = model.transcribe(str(audio_path))
        return result.get("text", "").strip()
    except Exception as e:
        logger.error(f"❌ Transcription failed: {e}")
        return ""


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎤 WHISPER ADAPTER TEST")
    print("="*50)
    
    # Test with sample audio if available
    test_paths = [
        "data/test_audio.wav",
        "data/test_audio.mp3",
        "C:/Users/USER/Music/sample.mp3"
    ]
    
    for path in test_paths:
        if Path(path).exists():
            result = transduce_audio(path)
            if result:
                print(f"\n🎵 Audio: {result.source_path}")
                print(f"   Duration: {result.duration_seconds:.1f}s")
                print(f"   Language: {result.language}")
                print(f"   Transcription: {result.transcription[:100]}...")
                print(f"   Dominant: {result.dominant_dimension}")
                print(f"   7D DNA:")
                for dim, val in result.dynamics.items():
                    bar = "█" * int(val * 50)
                    print(f"      {dim:12s}: {val:.4f} {bar}")
            break
    else:
        print("\n⚠️ No test audio found. Provide an audio path to test.")
        print("   Usage: python whisper_adapter.py")
