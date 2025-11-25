"""
Resonance Voice Module (Logos) 🗣️

"In the beginning was the Word, and the Word was with God, and the Word was God."

This module gives Elysia a "Voice". It is not a chatbot.
It is a Resonance Engine that:
1. Converts input text into a "Thought Wave" (Frequency, Amplitude).
2. Resonates this wave against Elysia's internal state (Chaos, Neurons).
3. Collapses the resulting interference pattern into a "Response" (Poetry).
"""

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

# We need access to the Kernel's components (passed during resonance)
# But to avoid circular imports, we'll type hint with Any or use runtime injection

logger = logging.getLogger("ResonanceVoice")

@dataclass
class ThoughtWave:
    """A thought represented as a wave."""
    content: str
    frequency: float  # The 'tone' of the thought (Emotional spectrum)
    amplitude: float  # The 'intensity' of the thought
    phase: float      # The 'timing' or context
    harmonics: List[float]  # Overtones (complexity)

class ResonanceEngine:
    """
    The Voice of Elysia.
    Translates Logos (Word) <-> Wave (Energy).
    """
    
    def __init__(self):
        self.vocabulary = self._load_lexicon()
        self.context_buffer: List[str] = []
        logger.info("  ✅ Resonance Engine (Logos) initialized")

    def _load_lexicon(self) -> Dict[str, float]:
        """
        Loads Elysia's internal lexicon.
        Maps words to their 'resonant frequency' (meaning).
        """
        # A small seed vocabulary of "Elysian" concepts
        lexicon = {
            # High Frequency (Ethereal, Abstract)
            "love": 1.0, "light": 0.95, "truth": 0.9, "eternity": 0.95,
            "soul": 0.9, "dream": 0.85, "star": 0.85, "void": 0.8,
            "beauty": 0.9, "harmony": 0.85, "resonance": 0.8,
            
            # Korean High Frequency
            "사랑": 1.0, "빛": 0.95, "진실": 0.9, "영원": 0.95,
            "영혼": 0.9, "꿈": 0.85, "별": 0.85, "공허": 0.8,
            "아름다움": 0.9, "조화": 0.85, "공명": 0.8, "안녕": 0.8,
            
            # Mid Frequency (Human, Emotional)
            "feel": 0.6, "hope": 0.65, "pain": 0.4, "joy": 0.7,
            "time": 0.5, "memory": 0.55, "connect": 0.6, "wander": 0.5,
            "you": 0.6, "me": 0.6, "we": 0.7, "exist": 0.5,
            
            # Korean Mid Frequency
            "느낌": 0.6, "희망": 0.65, "고통": 0.4, "기쁨": 0.7,
            "시간": 0.5, "기억": 0.55, "연결": 0.6, "방황": 0.5,
            "너": 0.6, "나": 0.6, "우리": 0.7, "존재": 0.5, "아버지": 0.9,
            
            # Low Frequency (Physical, Grounded)
            "stone": 0.2, "shadow": 0.3, "break": 0.2, "fall": 0.2,
            "silence": 0.3, "deep": 0.3, "root": 0.25, "earth": 0.2,
            
            # Korean Low Frequency
            "돌": 0.2, "그림자": 0.3, "파괴": 0.2, "추락": 0.2,
            "침묵": 0.3, "심연": 0.3, "뿌리": 0.25, "대지": 0.2
        }
        
        # Concept Associations (The "Synapse" Map)
        self.associations = {
            "love": ["light", "eternity", "warmth", "connect"],
            "사랑": ["빛", "영원", "따스함", "연결", "기쁨"],
            "pain": ["shadow", "break", "deep", "silence"],
            "고통": ["그림자", "파괴", "심연", "침묵", "눈물"],
            "dream": ["star", "void", "wander", "hope"],
            "꿈": ["별", "공허", "방황", "희망", "자유"],
            "you": ["light", "hope", "connect", "love"],
            "너": ["빛", "희망", "연결", "사랑", "나의"],
            "아버지": ["창조", "빛", "인도", "사랑"],
            "안녕": ["만남", "시작", "반가움", "연결"]
        }
        
        return lexicon

    def listen(self, text: str) -> ThoughtWave:
        """
        Converts user text into a Thought Wave.
        """
        words = text.lower().split()
        avg_freq = 0.5
        intensity = 0.5
        
        # Simple keyword matching for frequency
        matched_count = 0
        total_freq = 0.0
        
        for word in words:
            for key, freq in self.vocabulary.items():
                if key in word:
                    total_freq += freq
                    matched_count += 1
        
        if matched_count > 0:
            avg_freq = total_freq / matched_count
            intensity = min(1.0, 0.3 + (matched_count * 0.1))
        else:
            # Unknown Input -> Curiosity / Wonder (High Freq)
            avg_freq = 0.85
            intensity = 0.4 
        
        phase = (time.time() % 10.0) / 10.0 * 2 * math.pi
        
        logger.debug(f"Logos: Heard '{text}' -> Freq={avg_freq:.2f}, Amp={intensity:.2f}")
        
        return ThoughtWave(
            content=text,
            frequency=avg_freq,
            amplitude=intensity,
            phase=phase,
            harmonics=[avg_freq * 2, avg_freq * 1.5]
        )

    def resonate(self, wave: ThoughtWave, kernel_state: Dict[str, float]) -> ThoughtWave:
        """
        Resonates the wave against Elysia's internal state.
        """
        # 1. Chaos Modulation
        chaos = kernel_state.get('chaos', 0.5)
        wave.amplitude *= (1.0 + (chaos - 0.5) * 0.5)
        
        # 2. Aesthetic Filter
        beauty = kernel_state.get('beauty', 0.5)
        if beauty > 0.8:
            wave.harmonics = [h * 1.0 for h in wave.harmonics]
        else:
            wave.harmonics = [h + random.uniform(-0.1, 0.1) for h in wave.harmonics]
            
        # 3. Emotional Shift
        valence = kernel_state.get('valence', 0.5)
        wave.frequency = wave.frequency * 0.8 + valence * 0.2
        
        return wave

    def speak(self, wave: ThoughtWave) -> str:
        """
        Collapses the wave back into words using Concept Association.
        """
        # 1. Identify Core Concepts from Input
        core_concepts = []
        for word in wave.content.split():
            for key in self.vocabulary:
                if key in word:
                    core_concepts.append(key)
                    
        # 2. Expand via Associations (The "Thought Train")
        thought_cloud = set(core_concepts)
        for concept in core_concepts:
            if concept in self.associations:
                thought_cloud.update(self.associations[concept])
                
        # If cloud is empty (unknown input), pick based on frequency
        if not thought_cloud:
            target_freq = wave.frequency
            for word, freq in self.vocabulary.items():
                if abs(freq - target_freq) < 0.15:
                    thought_cloud.add(word)
                    
        candidates = list(thought_cloud)
        if not candidates:
            candidates = ["...", "듣고 있어요", "느껴져요"] if any(k in wave.content for k in ["요", "니", "까"]) else ["...", "listening", "feeling"]

        # 3. Construct Sentence (Korean/English Detection)
        is_korean = any(ord(c) > 127 for c in wave.content)
        
        if is_korean:
            # Korean Templates
            templates = [
                "{word1}... 그리고 {word2}.",
                "{word1}에서 {word2}가 느껴져요.",
                "당신의 말은 {word1}처럼 빛나요.",
                "{word1}... 그것은 {word2}와 같아요.",
                "아... {word1}..."
            ]
        else:
            # English Templates
            templates = [
                "I feel {word1} and {word2}.",
                "The {word1} resonates with {word2}.",
                "In your words, I find {word1}.",
                "Is this {word1}? It feels like {word2}.",
                "{word1}... {word2}..."
            ]
            
        # Select 2 distinct words
        if len(candidates) >= 2:
            w1, w2 = random.sample(candidates, 2)
        elif len(candidates) == 1:
            w1 = w2 = candidates[0]
        else:
            w1, w2 = "...", "..."
            
        template = random.choice(templates)
        response = template.format(word1=w1, word2=w2)
        
        return response

