"""
Text Wave Converter (텍스트 ⟷ 파동 변환기)
==========================================

"모든 언어는 파동이다. 의미는 주파수다."

This module converts text to wave representations and vice versa.
- Words → Frequencies (semantic similarity = frequency proximity)
- Sentences → Wave superposition (interference patterns)

[NEW 2025-12-15] Created as part of Phase 2: Transducers
"""

import logging
import math
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

from Core.Foundation.Wave.wave_tensor import WaveTensor

logger = logging.getLogger("TextWaveConverter")

# Semantic frequency bands (의미적 주파수 대역)
SEMANTIC_BANDS = {
    # Emotions (감정)
    "love": 528.0,      # Solfeggio MI - Love frequency
    "hope": 852.0,      # Solfeggio LA - Intuition  
    "joy": 639.0,       # Solfeggio FA - Connection
    "peace": 432.0,     # Universal harmony
    "fear": 174.0,      # Low frequency - Tension
    "anger": 285.0,     # Release frequency
    "sadness": 396.0,   # Solfeggio UT - Liberation
    
    # Concepts (개념)
    "truth": 528.0,     # Aligned with Love
    "beauty": 639.0,    # Harmony
    "good": 741.0,      # Awakening intuition
    "wisdom": 963.0,    # Solfeggio - Enlightenment
    
    # Actions (행위)
    "create": 417.0,    # Solfeggio RE - Change
    "destroy": 285.0,   # Low dissonance
    "grow": 396.0,      # Liberation and growth
    "learn": 741.0,     # Awakening
    
    # Elements (요소)
    "light": 963.0,     # Highest frequency
    "dark": 174.0,      # Lowest frequency
    "water": 432.0,     # Flow
    "fire": 852.0,      # Intensity
}




class TextWaveConverter:
    """
    The Transducer: Text ⟷ Wave
    
    변환기: 텍스트와 파동 사이의 다리
    
    Core principles:
    1. Semantic similarity = Frequency proximity
    2. Sentences = Wave interference patterns
    3. Meaning emerges from resonance
    """
    
    def __init__(self):
        self.semantic_bands = SEMANTIC_BANDS.copy()
        self.word_cache: Dict[str, WaveTensor] = {}
        
        # GlobalHub integration
        self._hub = None
        try:
            from Core.Intelligence.Consciousness.Ether.global_hub import get_global_hub
            self._hub = get_global_hub()
            self._hub.register_module(
                "TextWaveConverter",
                "Core/Foundation/text_wave_converter.py",
                ["text", "wave", "converter", "transducer", "language"],
                "Converts text to wave representations and back"
            )
            logger.info("   ✅ TextWaveConverter connected to GlobalHub")
        except ImportError:
            logger.warning("   ⚠️ GlobalHub not available")
        
        logger.info("🌊 TextWaveConverter initialized")
    
    def word_to_wave(self, word: str) -> WaveTensor:
        """
        Convert a single word to a WaveTensor.
        
        Strategy:
        1. Check semantic bands for known words
        2. Use phonetic analysis for unknown words
        3. Generate harmonics based on word structure
        
        Args:
            word: The word to convert
            
        Returns:
            WaveTensor representation
        """
        word_lower = word.lower().strip()
        
        # Check cache
        if word_lower in self.word_cache:
            return self.word_cache[word_lower]
        
        # Check semantic bands
        if word_lower in self.semantic_bands:
            freq = self.semantic_bands[word_lower]
        else:
            # Generate frequency from word structure
            freq = self._compute_frequency(word_lower)
        
        # Compute amplitude from word length (longer = more weight)
        amplitude = min(1.0, 0.3 + len(word_lower) * 0.1)
        
        # Compute phase from first character
        phase = (ord(word_lower[0]) % 12) * (math.pi / 6)  # 0 to 2π in 12 steps
        
        # Generate harmonics based on syllables (approximated)
        harmonics = self._generate_harmonics(word_lower, freq)
        
        # Create WaveTensor
        wt = WaveTensor(f"Word('{word}')")
        wt.add_component(freq, amplitude, phase)
        
        # Add harmonics
        for i, h_freq in enumerate(harmonics):
            h_amp = amplitude / (i + 2)
            wt.add_component(h_freq, h_amp, phase)

        self.word_cache[word_lower] = wt
        return wt
    
    def _compute_frequency(self, word: str) -> float:
        """
        Compute frequency for an unknown word.
        
        Uses phonetic heuristics:
        - Vowels tend to lower frequency (resonant)
        - Consonants tend to higher frequency (sharp)
        - Word energy is based on structure
        """
        base = 432.0  # Start from universal harmony
        
        # Vowel/consonant ratio
        vowels = set('aeiouAEIOU')
        vowel_count = sum(1 for c in word if c in vowels)
        consonant_count = len(word) - vowel_count
        
        if len(word) > 0:
            vowel_ratio = vowel_count / len(word)
        else:
            vowel_ratio = 0.5
        
        # More vowels = lower, warmer frequency
        # More consonants = higher, sharper frequency
        freq_shift = (0.5 - vowel_ratio) * 400  # ±200 Hz shift
        
        # Add hash-based uniqueness
        hash_val = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
        hash_shift = (hash_val % 200) - 100  # ±100 Hz
        
        frequency = base + freq_shift + hash_shift
        
        # Clamp to human hearing range subset
        return max(100.0, min(2000.0, frequency))
    
    def _generate_harmonics(self, word: str, base_freq: float) -> List[float]:
        """
        Generate harmonic frequencies based on word structure.
        
        Syllables create harmonics at integer multiples.
        """
        # Estimate syllable count (rough approximation)
        vowels = 'aeiouAEIOU'
        syllables = max(1, sum(1 for i, c in enumerate(word) 
                               if c in vowels and (i == 0 or word[i-1] not in vowels)))
        
        harmonics = []
        for i in range(1, min(syllables + 1, 5)):  # Up to 4 harmonics
            harmonics.append(base_freq * (i + 1))
        
        return harmonics
    
    def sentence_to_wave(self, sentence: str) -> WaveTensor:
        """
        Convert a sentence to a superposed WaveTensor.
        
        "문장은 의미의 간섭 패턴이다"
        
        Args:
            sentence: The sentence to convert
            
        Returns:
            WaveTensor: The superposition of all word waves
        """
        # Tokenize (simple split, could use better tokenization)
        words = [w for w in sentence.split() if w.strip()]
        
        if not words:
            return WaveTensor(f"EmptySentence")

        # Start with the first word
        result_wave = self.word_to_wave(words[0])
        
        # Superpose the rest
        for i in range(1, len(words)):
            next_wave = self.word_to_wave(words[i])
            result_wave = result_wave.superpose(next_wave)

        result_wave.name = f"Sentence('{sentence[:20]}...')"
        return result_wave
    
    def add_semantic_mapping(self, word: str, frequency: float):
        """
        Add a new semantic frequency mapping.
        
        Args:
            word: The word
            frequency: Its semantic frequency
        """
        self.semantic_bands[word.lower()] = frequency
        # Clear cache for this word
        if word.lower() in self.word_cache:
            del self.word_cache[word.lower()]
    
    def compute_resonance(self, wave1: WaveTensor, wave2: WaveTensor) -> float:
        """
        Compute resonance between two WaveTensors.
        
        Delegates to WaveTensor.resonance()
        """
        return wave1.resonance(wave2)
    
    def wave_to_text_descriptor(self, wave: WaveTensor) -> Dict:
        """
        Generate a text description of a wave's characteristics.
        
        파동의 특성을 언어로 표현
        """
        # Identify dominant frequency
        dominant_freq = 0.0
        if wave.active_frequencies.size > 0:
            idx = np.argmax(np.abs(wave._amplitudes))
            dominant_freq = wave.active_frequencies[idx]

        # Find closest semantic band
        closest_meaning = "unknown"
        min_diff = float('inf')
        
        for meaning, freq in self.semantic_bands.items():
            diff = abs(dominant_freq - freq)
            if diff < min_diff:
                min_diff = diff
                closest_meaning = meaning
        
        total_energy = wave.total_energy

        # Characterize energy level
        if total_energy > 5:
            energy_desc = "강렬한 (intense)"
        elif total_energy > 2:
            energy_desc = "활발한 (active)"
        else:
            energy_desc = "차분한 (calm)"
        
        return {
            "dominant_meaning": closest_meaning,
            "dominant_frequency": dominant_freq,
            "energy_level": energy_desc,
            "total_energy": total_energy
        }


# Singleton accessor
_converter = None

def get_text_wave_converter() -> TextWaveConverter:
    """Get or create the TextWaveConverter singleton."""
    global _converter
    if _converter is None:
        _converter = TextWaveConverter()
    return _converter


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    converter = get_text_wave_converter()
    
    print("\n" + "="*60)
    print("🌊 Text Wave Converter Test")
    print("="*60)
    
    # Test single word
    word = "love"
    wave = converter.word_to_wave(word)
    print(f"\n단어 '{word}':")
    print(f"  {wave}")
    
    # Test sentence
    sentence1 = "I love you"
    sentence2 = "I hate you"
    
    wave1 = converter.sentence_to_wave(sentence1)
    wave2 = converter.sentence_to_wave(sentence2)
    
    print(f"\n문장 '{sentence1}':")
    desc1 = converter.wave_to_text_descriptor(wave1)
    print(f"  지배 의미: {desc1['dominant_meaning']}")
    print(f"  지배 주파수: {desc1['dominant_frequency']:.1f} Hz")
    print(f"  에너지: {desc1['energy_level']}")
    
    print(f"\n문장 '{sentence2}':")
    desc2 = converter.wave_to_text_descriptor(wave2)
    print(f"  지배 의미: {desc2['dominant_meaning']}")
    print(f"  지배 주파수: {desc2['dominant_frequency']:.1f} Hz")
    
    # Test resonance
    resonance = converter.compute_resonance(wave1, wave2)
    print(f"\n공명 점수 ('{sentence1}' vs '{sentence2}'): {resonance:.3f}")
    
    # Test Korean
    korean = "사랑해요"
    wave_kr = converter.sentence_to_wave(korean)
    desc_kr = converter.wave_to_text_descriptor(wave_kr)
    print(f"\n한국어 '{korean}':")
    print(f"  지배 주파수: {desc_kr['dominant_frequency']:.1f} Hz")
    print(f"  에너지: {desc_kr['energy_level']}")
    
    print("\n" + "="*60)
    print("✅ TextWaveConverter 테스트 완료")
    print("="*60)
