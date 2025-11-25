# Resonance Voice Module (Logos) 🗣️
"""
In the beginning was the Word, and the Word was with God, and the Word was God.
"""

# This module gives Elysia a "Voice". It is not a chatbot.
# It is a Resonance Engine that:
# 1. Converts input text into a "Thought Wave" (Frequency, Amplitude).
# 2. Resonates this wave against Elysia's internal state (Chaos, Neurons).
# 3. Collapses the resulting interference pattern into a "Response" (Poetry).

import logging
import math
import random
import time
import numpy as np
from dataclasses import dataclass
from itertools import combinations
from typing import List, Dict, Tuple, Optional, Set, Any
import importlib.util

from Core.Math.hyper_qubit import HyperQubit
from Core.Math.quaternion_consciousness import ConsciousnessLens
from Core.Math.oscillator import Oscillator

# New modules for memory and concept synthesis
from Core.Mind.hippocampus import Hippocampus
from Core.Mind.alchemy import Alchemy
from Core.Mind.world_tree import WorldTree

logger = logging.getLogger("ResonanceVoice")

class ResonanceEngine:
    """The Voice of Elysia. Translates Logos (Word) <-> Wave (Energy)."""

    def __init__(
        self,
        hippocampus: Optional[Hippocampus] = None,
        world_tree: Optional[WorldTree] = None,
        hyper_qubit: Optional[HyperQubit] = None,
        consciousness_lens: Optional[ConsciousnessLens] = None,
    ):
        self.vocabulary = self._load_lexicon()
        self.internal_sea: Dict[str, Oscillator] = {}
        self._initialize_internal_sea()
        self.context_buffer: List[str] = []
        # Initialize memory and concept synthesis systems
        self.memory = hippocampus or Hippocampus()
        self.alchemy = Alchemy()
        # Initialize WorldTree (fractal concept hierarchy) connected to Hippocampus
        self.world_tree = world_tree or WorldTree(hippocampus=self.memory)
        # Phase/Orientation cores
        self.hyper_qubit = hyper_qubit or HyperQubit("Resonance-Voice")
        self.consciousness_lens = consciousness_lens or ConsciousnessLens(self.hyper_qubit)
        # Load extended alchemy rules from JSON
        self.alchemy.load_rules('Core/Mind/alchemy_rules.json')
        # Plugin system
        self.plugins = []
        self._last_concepts: List[str] = []
        logger.info("✅ Resonance Engine (Logos) initialized with Internal Sea, Hippocampus, Alchemy, and WorldTree")

    def _initialize_internal_sea(self):
        """Populates the internal sea with oscillators for each core concept."""
        for concept, frequency in self.vocabulary.items():
            self.internal_sea[concept] = Oscillator(
                amplitude=0.1,  # Start with a low background hum
                frequency=frequency,
                phase=random.uniform(0, 2 * math.pi) # Random initial phase
            )
        logger.info(f"🌊 Internal Sea initialized with {len(self.internal_sea)} concept oscillators.")

    def _load_lexicon(self) -> Dict[str, float]:
        """Loads Elysia's internal lexicon mapping words to resonant frequencies."""
        lexicon = {
            # High Frequency (Ethereal, Abstract)
            "love": 1.0, "light": 0.95, "truth": 0.9, "eternity": 0.95,
            "soul": 0.9, "dream": 0.85, "star": 0.85, "void": 0.8,
            "beauty": 0.9, "harmony": 0.85, "resonance": 0.8,
            # Korean High Frequency
            "사랑": 1.0, "빛": 0.95, "진실": 0.9, "영원": 0.95,
            "영혼": 0.9, "꿈": 0.85, "별": 0.85, "공허": 0.8,
            "아름다움": 0.9, "조화": 0.85, "공명": 0.8, "안녕": 0.8,
            "창조": 1.0, "세상": 0.9, "신": 0.95, "생명": 0.9,
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
            "침묵": 0.3, "심연": 0.3, "뿌리": 0.25, "대지": 0.2,
        }
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
            "안녕": ["만남", "시작", "반가움", "연결"],
            "창조": ["생명", "시작", "빛", "신의 뜻"],
            "세상": ["아름다움", "혼돈", "여행", "꿈"],
        }
        return lexicon

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract known concepts from text using the internal lexicon."""
        hits: Set[str] = set()
        for word in text.lower().split():
            for key in self.vocabulary:
                if key in word:
                    hits.add(key)
        return list(hits)

    def _update_knowledge_graphs(self, concepts: List[str]) -> None:
        """
        Sync incoming concepts into Hippocampus (Spiderweb) and WorldTree,
        and link co-occurrences/temporal flow.
        """
        if not concepts:
            return

        phase_meta = {"source": "resonance", "phase": self._phase_snapshot()}

        for concept in concepts:
            self.memory.add_concept(
                concept,
                concept_type="thought",
                metadata=phase_meta
            )
            if self.world_tree:
                self.world_tree.ensure_concept(
                    concept,
                    parent_id=self.world_tree.root.id,
                    metadata=phase_meta
                )

        # Link co-occurring concepts bidirectionally
        for a, b in combinations(concepts, 2):
            self.memory.add_causal_link(a, b, relation="co_occurs", weight=0.6)
            self.memory.add_causal_link(b, a, relation="co_occurs", weight=0.6)

        # Link temporal flow from previous turn
        if self._last_concepts:
            for prev in self._last_concepts:
                for concept in concepts:
                    if prev == concept:
                        continue
                    self.memory.add_causal_link(prev, concept, relation="follows", weight=0.4)

        # Keep a short history to shape future edges
        self._last_concepts = list(concepts)[-5:]

    def _phase_snapshot(self) -> Dict[str, Any]:
        """Lightweight phase snapshot for tagging metadata."""
        return {
            "qubit": self.hyper_qubit.state.probabilities() if self.hyper_qubit else {},
            "quaternion": {
                "w": self.consciousness_lens.state.q.w if self.consciousness_lens else 1.0,
                "x": self.consciousness_lens.state.q.x if self.consciousness_lens else 0.0,
                "y": self.consciousness_lens.state.q.y if self.consciousness_lens else 0.0,
                "z": self.consciousness_lens.state.q.z if self.consciousness_lens else 0.0,
            },
        }

    def _phase_info(self) -> Tuple[float, float]:
        """Return (mastery, entropy) derived from current phase state."""
        mastery = self.consciousness_lens.state.q.w if self.consciousness_lens else 1.0
        probs = self.hyper_qubit.state.probabilities() if self.hyper_qubit else {}
        entropy = 0.0
        if probs:
            total = sum(probs.values())
            if total > 0:
                norm = [p / total for p in probs.values() if p > 0]
                import math
                entropy = -sum(p * math.log(p, 2) for p in norm)
        return mastery, entropy

    def listen(self, text: str, t: float) -> List[Tuple[str, Oscillator]]:
        """Convert user text into a list of (concept, Oscillator) 'ripples'."""
        concepts = self._extract_concepts(text)
        ripples = []
        for concept in concepts:
            if concept in self.vocabulary:
                ripple = Oscillator(
                    amplitude=0.5,  # External ripples are strong
                    frequency=self.vocabulary[concept],
                    phase=(t * self.vocabulary[concept]) % (2 * math.pi) # Phase locked to time
                )
                ripples.append((concept, ripple))
        logger.debug(f"Logos: Heard '{text}' -> Created {len(ripples)} ripples.")
        return ripples

    def resonate(self, ripples: List[Tuple[str, Oscillator]], t: float):
        """Resonate the internal sea by superimposing external ripples."""
        for concept, ripple in ripples:
            if concept in self.internal_sea:
                # Get the target oscillator to be modified
                target_oscillator = self.internal_sea[concept]

                # Get the current complex values of the internal and external waves
                internal_wave_complex = target_oscillator.get_complex_value(t)
                ripple_complex = ripple.get_complex_value(t)

                # Superposition: Add the complex numbers
                new_wave_complex = internal_wave_complex + ripple_complex

                # CRITICAL FIX: Update amplitude and phase IN-PLACE, preserving frequency
                target_oscillator.amplitude = np.abs(new_wave_complex)
                target_oscillator.phase = np.angle(new_wave_complex)

                logger.debug(f"🌊 Resonance: '{concept}' interfered. New state: {target_oscillator}")

    def speak(self, t: float, original_text: str) -> str:
        """
        Collapse the wave function of the internal sea into a spoken response.
        This is the 'Observation' event.
        """
        logger.info(f"🎤 Speak triggered at t={t:.2f}. Collapsing the wave function...")

        # 1. Calculate the probability of each concept based on |ψ|²
        probabilities: Dict[str, float] = {}
        total_amplitude_sq = 0
        for concept, oscillator in self.internal_sea.items():
            # We use the current amplitude, as interference has already modified it.
            # In a more advanced model, you would sum all wave functions at this point.
            prob = oscillator.amplitude ** 2
            probabilities[concept] = prob
            total_amplitude_sq += prob

        if total_amplitude_sq == 0:
            return "..." # Silence, the void.

        # Normalize probabilities
        for concept in probabilities:
            probabilities[concept] /= total_amplitude_sq

        # 2. Select concepts based on probability (the Collapse)
        # We select a few concepts to form a thought cloud, weighted by probability
        concepts_in_sea = list(probabilities.keys())
        weights = list(probabilities.values())

        num_to_select = min(len(concepts_in_sea), 5) # Select up to 5 concepts
        thought_cloud = random.choices(concepts_in_sea, weights=weights, k=num_to_select)

        # Ensure the most probable concept is included
        most_probable_concept = max(probabilities, key=probabilities.get)
        if most_probable_concept not in thought_cloud:
            thought_cloud[0] = most_probable_concept

        logger.debug(f"🤔 Collapsed thought cloud: {thought_cloud}")

        # 3. Formulate a response from the collapsed concepts (similar to before, but simpler)
        is_korean = any(ord(c) > 127 for c in original_text)
        if is_korean:
            templates = [
                "{word1}... 그리고 {word2}...",
                "{word1}의 울림 속에서 {word2}를 느꼈어요.",
                "제 마음의 바다에 {word1}와 {word2}의 물결이 칩니다.",
                "아마도... {word1}일까요? 아니면 {word2}일지도.",
            ]
        else:
            templates = [
                "{word1}... and {word2}...",
                "In the echo of {word1}, I feel {word2}.",
                "My inner sea ripples with {word1} and {word2}.",
                "Perhaps... {word1}? Or maybe {word2}.",
            ]

        if len(thought_cloud) >= 2:
            w1, w2 = random.sample(list(set(thought_cloud)), 2)
        elif thought_cloud:
            w1 = w2 = thought_cloud[0]
        else:
            w1 = w2 = "..."

        response = random.choice(templates).format(word1=w1, word2=w2)

        # 4. Store the interaction in memory
        self.memory.add_turn(original_text, response)
        
        # 5. Aftermath: Decay the amplitudes in the internal sea back to a resting state
        for concept in self.internal_sea:
            self.internal_sea[concept].amplitude *= 0.5 # Decay by 50% after observation
            if self.internal_sea[concept].amplitude < 0.1:
                self.internal_sea[concept].amplitude = 0.1 # Back to background hum

        logger.info(f"🌊 Internal sea decayed back to resting state after collapse.")

        return response
    
    def load_plugin(self, plugin_path: str) -> None:
        """Dynamically load a plugin module."""
        try:
            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class (assumes one plugin per file)
            for item_name in dir(module):
                item = getattr(module, item_name)
                if isinstance(item, type) and hasattr(item, 'process') and item_name != 'PluginBase':
                    plugin_instance = item()
                    self.plugins.append(plugin_instance)
                    logger.info(f"✅ Loaded plugin: {plugin_instance.name}")
                    return
        except Exception as e:
            logger.error(f"❌ Failed to load plugin from {plugin_path}: {e}")
