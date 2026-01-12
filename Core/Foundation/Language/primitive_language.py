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


# Basic Phoneme Library (Inspired by Universal Phonetics)
PHONEME_LIBRARY = {
    # Vowels (high openness)
    'a': PhonemeRotor('a', 432.0, openness=1.0, brightness=0.7, softness=1.0),
    'e': PhonemeRotor('e', 528.0, openness=0.8, brightness=0.9, softness=1.0),
    'i': PhonemeRotor('i', 639.0, openness=0.6, brightness=1.0, softness=1.0),
    'o': PhonemeRotor('o', 396.0, openness=0.9, brightness=0.3, softness=1.0),
    'u': PhonemeRotor('u', 285.0, openness=0.7, brightness=0.1, softness=1.0),
    
    # Consonants - Stops (low softness)
    'k': PhonemeRotor('k', 852.0, openness=0.0, brightness=0.5, softness=0.0),
    't': PhonemeRotor('t', 741.0, openness=0.0, brightness=0.7, softness=0.1),
    'p': PhonemeRotor('p', 396.0, openness=0.0, brightness=0.4, softness=0.1),
    
    # Consonants - Nasals (high softness)
    'm': PhonemeRotor('m', 174.0, openness=0.0, brightness=0.3, softness=0.9),
    'n': PhonemeRotor('n', 264.0, openness=0.1, brightness=0.5, softness=0.9),
    
    # Consonants - Fricatives
    's': PhonemeRotor('s', 963.0, openness=0.1, brightness=0.9, softness=0.3),
    'f': PhonemeRotor('f', 639.0, openness=0.1, brightness=0.6, softness=0.4),
    
    # Consonants - Liquids
    'l': PhonemeRotor('l', 396.0, openness=0.3, brightness=0.6, softness=0.8),
    'r': PhonemeRotor('r', 432.0, openness=0.2, brightness=0.5, softness=0.6),
}


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
        """Convert text to phoneme sequence."""
        for char in self.text.lower():
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
