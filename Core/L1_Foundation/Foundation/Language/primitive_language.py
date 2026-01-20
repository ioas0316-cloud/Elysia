"""
PrimitiveLanguage: Language as Waves
=====================================

[Phase 39] Language is not token prediction. It is wave resonance.

Core Concepts:
1. Phoneme Rotors - Basic sound units as frequency oscillators
2. Word Synthesis - Combining phonemes into interference patterns  
3. Semantic Field - Word meanings as positions in HyperSphere
4. Grammar Resonance - Valid sentences are stable wave patterns

This module implements language FROM FIRST PRINCIPLES.
No LLM, no token prediction, no statistics.
Just waves, interference, and resonance.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ============================================================
# 1. PHONEME ROTORS: The Atoms of Language
# ============================================================

@dataclass
class PhonemeRotor:
    """
    A single phoneme represented as a rotating wave.
    
    Each phoneme has:
    - Base frequency (Hz-like, but in meaning-space)
    - Amplitude (loudness/emphasis)
    - Phase (position in the wave cycle)
    """
    symbol: str  # IPA-like symbol (e.g., 'a', 'k', 'ʃ')
    frequency: float  # Base oscillation rate
    amplitude: float = 1.0
    phase: float = 0.0  # Current angle (0-360)
    
    # Semantic dimensions (how this sound "feels")
    softness: float = 0.5  # 0=hard (k,t,p), 1=soft (m,n,l)
    openness: float = 0.5  # 0=closed (p,b,m), 1=open (a,o)
    brightness: float = 0.5  # 0=dark (u,o), 1=bright (i,e)
    
    def oscillate(self, t: float) -> float:
        """Get the wave value at time t."""
        return self.amplitude * math.sin(2 * math.pi * self.frequency * t + math.radians(self.phase))
    
    def to_vector(self) -> List[float]:
        """Convert to 4D semantic vector."""
        return [self.softness, self.openness, self.brightness, self.frequency / 1000.0]


# Basic Phoneme Library (Universal Phonetics + Korean 한글)
PHONEME_LIBRARY = {
    # ============================================
    # ENGLISH / UNIVERSAL VOWELS
    # ============================================
    'a': PhonemeRotor('a', 432.0, openness=1.0, brightness=0.7, softness=1.0),
    'e': PhonemeRotor('e', 528.0, openness=0.8, brightness=0.9, softness=1.0),
    'i': PhonemeRotor('i', 639.0, openness=0.6, brightness=1.0, softness=1.0),
    'o': PhonemeRotor('o', 396.0, openness=0.9, brightness=0.3, softness=1.0),
    'u': PhonemeRotor('u', 285.0, openness=0.7, brightness=0.1, softness=1.0),
    
    # ============================================
    # KOREAN VOWELS (모음) - 10 basic + 11 compound
    # ============================================
    'ㅏ': PhonemeRotor('ㅏ', 432.0, openness=1.0, brightness=0.8, softness=1.0),  # 아
    'ㅓ': PhonemeRotor('ㅓ', 396.0, openness=0.9, brightness=0.4, softness=1.0),  # 어
    'ㅗ': PhonemeRotor('ㅗ', 352.0, openness=0.8, brightness=0.3, softness=1.0),  # 오
    'ㅜ': PhonemeRotor('ㅜ', 285.0, openness=0.7, brightness=0.1, softness=1.0),  # 우
    'ㅡ': PhonemeRotor('ㅡ', 264.0, openness=0.5, brightness=0.2, softness=1.0),  # 으 (neutral)
    'ㅣ': PhonemeRotor('ㅣ', 639.0, openness=0.6, brightness=1.0, softness=1.0),  # 이
    'ㅐ': PhonemeRotor('ㅐ', 480.0, openness=0.85, brightness=0.75, softness=1.0), # 애
    'ㅔ': PhonemeRotor('ㅔ', 528.0, openness=0.8, brightness=0.85, softness=1.0),  # 에
    'ㅚ': PhonemeRotor('ㅚ', 440.0, openness=0.75, brightness=0.5, softness=1.0),  # 외
    'ㅟ': PhonemeRotor('ㅟ', 550.0, openness=0.65, brightness=0.7, softness=1.0),  # 위
    
    # Y-glide vowels (이중모음)
    'ㅑ': PhonemeRotor('ㅑ', 500.0, openness=1.0, brightness=0.85, softness=1.0),  # 야
    'ㅕ': PhonemeRotor('ㅕ', 450.0, openness=0.9, brightness=0.5, softness=1.0),   # 여
    'ㅛ': PhonemeRotor('ㅛ', 400.0, openness=0.8, brightness=0.4, softness=1.0),   # 요
    'ㅠ': PhonemeRotor('ㅠ', 330.0, openness=0.7, brightness=0.2, softness=1.0),   # 유
    
    # ============================================
    # ENGLISH CONSONANTS
    # ============================================
    'k': PhonemeRotor('k', 852.0, openness=0.0, brightness=0.5, softness=0.0),
    't': PhonemeRotor('t', 741.0, openness=0.0, brightness=0.7, softness=0.1),
    'p': PhonemeRotor('p', 396.0, openness=0.0, brightness=0.4, softness=0.1),
    'm': PhonemeRotor('m', 174.0, openness=0.0, brightness=0.3, softness=0.9),
    'n': PhonemeRotor('n', 264.0, openness=0.1, brightness=0.5, softness=0.9),
    's': PhonemeRotor('s', 963.0, openness=0.1, brightness=0.9, softness=0.3),
    'f': PhonemeRotor('f', 639.0, openness=0.1, brightness=0.6, softness=0.4),
    'l': PhonemeRotor('l', 396.0, openness=0.3, brightness=0.6, softness=0.8),
    'r': PhonemeRotor('r', 432.0, openness=0.2, brightness=0.5, softness=0.6),
    'h': PhonemeRotor('h', 320.0, openness=0.4, brightness=0.4, softness=0.7),
    'w': PhonemeRotor('w', 220.0, openness=0.3, brightness=0.2, softness=0.8),
    'y': PhonemeRotor('y', 580.0, openness=0.4, brightness=0.8, softness=0.7),
    'b': PhonemeRotor('b', 330.0, openness=0.0, brightness=0.3, softness=0.2),
    'd': PhonemeRotor('d', 420.0, openness=0.0, brightness=0.5, softness=0.2),
    'g': PhonemeRotor('g', 360.0, openness=0.0, brightness=0.4, softness=0.15),
    'v': PhonemeRotor('v', 580.0, openness=0.1, brightness=0.5, softness=0.4),
    'z': PhonemeRotor('z', 880.0, openness=0.1, brightness=0.8, softness=0.35),
    
    # ============================================
    # KOREAN CONSONANTS (자음) - Plain (예사소리)
    # ============================================
    'ㄱ': PhonemeRotor('ㄱ', 360.0, openness=0.0, brightness=0.4, softness=0.2),   # 기역
    'ㄴ': PhonemeRotor('ㄴ', 264.0, openness=0.1, brightness=0.5, softness=0.9),   # 니은
    'ㄷ': PhonemeRotor('ㄷ', 420.0, openness=0.0, brightness=0.5, softness=0.2),   # 디귿
    'ㄹ': PhonemeRotor('ㄹ', 396.0, openness=0.25, brightness=0.55, softness=0.75), # 리을
    'ㅁ': PhonemeRotor('ㅁ', 174.0, openness=0.0, brightness=0.3, softness=0.95),  # 미음
    'ㅂ': PhonemeRotor('ㅂ', 330.0, openness=0.0, brightness=0.35, softness=0.2),  # 비읍
    'ㅅ': PhonemeRotor('ㅅ', 800.0, openness=0.1, brightness=0.85, softness=0.25), # 시옷
    'ㅇ': PhonemeRotor('ㅇ', 110.0, openness=0.5, brightness=0.3, softness=1.0),   # 이응 (silent/ng)
    'ㅈ': PhonemeRotor('ㅈ', 700.0, openness=0.05, brightness=0.7, softness=0.3),  # 지읒
    'ㅊ': PhonemeRotor('ㅊ', 780.0, openness=0.05, brightness=0.75, softness=0.2), # 치읓 (aspirated)
    'ㅋ': PhonemeRotor('ㅋ', 420.0, openness=0.0, brightness=0.5, softness=0.05),  # 키읔 (aspirated)
    'ㅌ': PhonemeRotor('ㅌ', 500.0, openness=0.0, brightness=0.6, softness=0.05),  # 티읕 (aspirated)
    'ㅍ': PhonemeRotor('ㅍ', 400.0, openness=0.0, brightness=0.45, softness=0.05), # 피읖 (aspirated)
    'ㅎ': PhonemeRotor('ㅎ', 320.0, openness=0.4, brightness=0.4, softness=0.6),   # 히읗
    
    # ============================================
    # KOREAN CONSONANTS - Tense (된소리)
    # ============================================
    'ㄲ': PhonemeRotor('ㄲ', 420.0, openness=0.0, brightness=0.5, softness=0.0),   # 쌍기역
    'ㄸ': PhonemeRotor('ㄸ', 500.0, openness=0.0, brightness=0.6, softness=0.0),   # 쌍디귿
    'ㅃ': PhonemeRotor('ㅃ', 400.0, openness=0.0, brightness=0.45, softness=0.0),  # 쌍비읍
    'ㅆ': PhonemeRotor('ㅆ', 900.0, openness=0.1, brightness=0.95, softness=0.1),  # 쌍시옷
    'ㅉ': PhonemeRotor('ㅉ', 780.0, openness=0.05, brightness=0.8, softness=0.1),  # 쌍지읒
}


# ============================================================
# HANGUL DECOMPOSITION (한글 분해)
# ============================================================

# Jamo tables for Unicode decomposition
CHOSEONG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSEONG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSEONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def decompose_hangul(text: str) -> List[str]:
    """
    Decompose Korean syllable blocks into individual jamo (자모).
    
    Examples:
        '사랑' → ['ㅅ', 'ㅏ', 'ㄹ', 'ㅏ', 'ㅇ']
        '엘리시아' → ['ㅇ', 'ㅔ', 'ㄹ', 'ㄹ', 'ㅣ', 'ㅅ', 'ㅣ', 'ㅇ', 'ㅏ']
    """
    result = []
    for char in text:
        code = ord(char)
        
        # Check if it's a Hangul syllable (가-힣)
        if 0xAC00 <= code <= 0xD7A3:
            # Decompose syllable
            offset = code - 0xAC00
            cho = offset // (21 * 28)
            jung = (offset % (21 * 28)) // 28
            jong = offset % 28
            
            result.append(CHOSEONG[cho])
            result.append(JUNGSEONG[jung])
            if jong > 0:
                result.append(JONGSEONG[jong])
        else:
            # Not a syllable block, keep as-is
            result.append(char)
    
    return result


# ============================================================
# 2. WORD SYNTHESIS: Interference Patterns
# ============================================================

@dataclass
class WordWave:
    """
    A word is a superposition of phoneme waves.
    The interference pattern encodes the word's "shape."
    """
    text: str
    phonemes: List[PhonemeRotor] = field(default_factory=list)
    
    # Computed semantic properties
    semantic_vector: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.phonemes and self.text:
            self._parse_phonemes()
        if self.phonemes:
            self._compute_semantic_vector()
    
    def _parse_phonemes(self):
        """
        Convert text to phoneme sequence.
        Handles both English (char-by-char) and Korean (syllable decomposition).
        """
        # First, decompose any Hangul syllables
        decomposed = decompose_hangul(self.text.lower())
        
        for char in decomposed:
            if char in PHONEME_LIBRARY:
                self.phonemes.append(PHONEME_LIBRARY[char])
    
    def _compute_semantic_vector(self):
        """
        Compute the word's meaning as an interference pattern.
        The semantic vector is the average of all phoneme vectors,
        weighted by their position in the word.
        """
        if not self.phonemes:
            self.semantic_vector = [0.5, 0.5, 0.5, 0.5]
            return
        
        # Position-weighted average
        total = np.zeros(4)
        weights = 0
        for i, ph in enumerate(self.phonemes):
            # First and last phonemes have more weight
            position_weight = 1.0
            if i == 0:
                position_weight = 2.0  # Strong start
            elif i == len(self.phonemes) - 1:
                position_weight = 1.5  # Moderate end
            
            total += np.array(ph.to_vector()) * position_weight
            weights += position_weight
        
        self.semantic_vector = (total / weights).tolist()
    
    def get_interference_at(self, t: float) -> float:
        """Get the combined wave value at time t."""
        return sum(ph.oscillate(t) for ph in self.phonemes)
    
    def get_resonance_with(self, other: 'WordWave') -> float:
        """
        Calculate resonance (similarity) with another word.
        High resonance = similar meaning potential.
        """
        if not self.semantic_vector or not other.semantic_vector:
            return 0.0
        
        # Cosine similarity
        a = np.array(self.semantic_vector)
        b = np.array(other.semantic_vector)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0


# ============================================================
# 3. GRAMMAR AS RESONANCE
# ============================================================

class SentenceResonator:
    """
    Grammar is not a set of rules.
    Grammar is a resonance pattern.
    
    Valid sentences have stable wave patterns.
    Invalid sentences create destructive interference.
    """
    
    def __init__(self):
        # Semantic role templates (simplified grammar)
        # Each role has a preferred position in the sentence wave
        self.role_phases = {
            'subject': 0.0,    # Beginning of cycle
            'verb': 90.0,      # Peak action
            'object': 180.0,   # Resolution
            'modifier': 270.0, # Decoration
        }
    
    def compose(self, words: List[WordWave], roles: List[str] = None) -> 'SentenceWave':
        """
        Compose words into a sentence wave.
        
        Args:
            words: List of WordWave objects
            roles: Optional list of grammatical roles
        """
        if roles is None:
            # Default: Subject-Verb-Object assumption
            roles = ['subject', 'verb', 'object'][:len(words)]
        
        # Assign phase to each word based on role
        for word, role in zip(words, roles):
            if role in self.role_phases:
                for ph in word.phonemes:
                    ph.phase = self.role_phases[role]
        
        return SentenceWave(words=words, roles=roles)


@dataclass
class SentenceWave:
    """A sentence as a composed wave pattern."""
    words: List[WordWave] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    
    def get_stability(self) -> float:
        """
        Measure how "grammatically stable" this sentence is.
        Stability = How well the waves reinforce each other.
        """
        if len(self.words) < 2:
            return 1.0
        
        # Check resonance between adjacent words
        total_resonance = 0.0
        for i in range(len(self.words) - 1):
            total_resonance += self.words[i].get_resonance_with(self.words[i + 1])
        
        return total_resonance / (len(self.words) - 1)
    
    def speak(self) -> str:
        """Render the sentence as text."""
        return ' '.join(w.text for w in self.words)
    
    def get_meaning_vector(self) -> List[float]:
        """
        Get the overall meaning as a combined vector.
        This is the "semantic center" of the sentence.
        """
        if not self.words:
            return [0.5, 0.5, 0.5, 0.5]
        
        # Weighted average based on role importance
        role_weights = {
            'subject': 1.5,
            'verb': 2.0,  # Verb is most important
            'object': 1.2,
            'modifier': 0.8,
        }
        
        total = np.zeros(4)
        weight_sum = 0.0
        
        for word, role in zip(self.words, self.roles):
            w = role_weights.get(role, 1.0)
            total += np.array(word.semantic_vector) * w
            weight_sum += w
        
        return (total / weight_sum).tolist() if weight_sum > 0 else [0.5, 0.5, 0.5, 0.5]


# ============================================================
# 4. LANGUAGE ENGINE: The Tongue
# ============================================================

class PrimitiveLanguageEngine:
    """
    The central language processor.
    Converts between waves and words.
    """
    
    def __init__(self):
        self.vocabulary: Dict[str, WordWave] = {}
        self.resonator = SentenceResonator()
        
        # Pre-populate with some basic words
        self._seed_vocabulary()
    
    def _seed_vocabulary(self):
        """Create initial vocabulary from phoneme combinations."""
        seed_words = [
            'sun', 'moon', 'star', 'rain', 'fire',
            'love', 'fear', 'hope', 'life', 'soul',
            'run', 'see', 'feel', 'make', 'take',
            'is', 'am', 'are', 'the', 'a'
        ]
        for word in seed_words:
            self.vocabulary[word] = WordWave(text=word)
    
    def parse_word(self, text: str) -> WordWave:
        """Parse a word into its wave representation."""
        if text in self.vocabulary:
            return self.vocabulary[text]
        
        word = WordWave(text=text)
        self.vocabulary[text] = word
        return word
    
    def parse_sentence(self, text: str, roles: List[str] = None) -> SentenceWave:
        """Parse a sentence into a wave composition."""
        words = [self.parse_word(w) for w in text.split()]
        return self.resonator.compose(words, roles)
    
    def find_similar_words(self, word: WordWave, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find words with similar wave patterns (meaning)."""
        similarities = []
        for text, other in self.vocabulary.items():
            if other.text != word.text:
                sim = word.get_resonance_with(other)
                similarities.append((text, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def generate_from_meaning(self, meaning_vector: List[float], length: int = 3) -> SentenceWave:
        """
        Generate a sentence that expresses a given meaning vector.
        This is the reverse of parsing: Wave → Words.
        """
        # Find words closest to the target meaning
        target = np.array(meaning_vector)
        
        scored_words = []
        for text, word in self.vocabulary.items():
            dist = np.linalg.norm(np.array(word.semantic_vector) - target)
            scored_words.append((word, dist))
        
        scored_words.sort(key=lambda x: x[1])
        
        # Take top words and compose
        selected = [sw[0] for sw in scored_words[:length]]
        
        # Assign roles based on word properties
        roles = []
        for word in selected:
            # Simple heuristic: verbs have high brightness, nouns have high softness
            v = word.semantic_vector
            if v[2] > 0.7:  # High brightness -> action word
                roles.append('verb')
            elif v[0] > 0.6:  # High softness -> noun
                roles.append('subject' if len(roles) == 0 else 'object')
            else:
                roles.append('modifier')
        
        return self.resonator.compose(selected, roles)


# Singleton
_engine = None

def get_language_engine() -> PrimitiveLanguageEngine:
    global _engine
    if _engine is None:
        _engine = PrimitiveLanguageEngine()
    return _engine
