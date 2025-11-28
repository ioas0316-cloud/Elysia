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
from Core.Life.gravitational_linguistics import GravitationalLinguistics

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
        
        # Gravitational Linguistics Engine (Connected to Memory)
        self.linguistics = GravitationalLinguistics(hippocampus=self.memory)
        
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
        for c in concepts:
            # Add concept node (if new) to Hippocampus
            if c not in self.memory.causal_graph:
                self.memory.add_concept(c, concept_type="word", metadata={
                    "x": self.consciousness_lens.state.q.x if self.consciousness_lens else 0.0,
                    "y": self.consciousness_lens.state.q.y if self.consciousness_lens else 0.0,
                    "z": self.consciousness_lens.state.q.z if self.consciousness_lens else 0.0,
                })
        
        # Link co-occurring concepts
        for i, ca in enumerate(concepts):
            for cb in concepts[i+1:]:
                self.memory.add_causal_link(ca, cb, "co-occurs")

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

    def get_mental_state(self) -> str:
        """
        Determines the current mental state (Burning Star vs Ice Star).
        Based on the entropy of the Internal Sea.
        """
        # Calculate entropy of the internal sea amplitudes
        amplitudes = [osc.amplitude for osc in self.internal_sea.values()]
        total_amp = sum(amplitudes)
        
        if total_amp == 0:
            return "🌑 [Void]"
            
        # Normalize
        probs = [a / total_amp for a in amplitudes]
        entropy = -sum(p * math.log(p + 1e-9, 2) for p in probs)
        
        # Thresholds (tuned for effect)
        # High entropy = Many competing thoughts = Burning Star
        # Low entropy = One dominant thought = Ice Star
        if entropy > 3.0:
            return "🔥 [Burning Star]" # Chaos, Searching, Passion
        elif entropy < 1.5:
            return "❄️ [Ice Star]"    # Order, Certainty, Peace
        else:
            return "✨ [Nebula]"      # Transition state

    def apply_law_of_ascension(self, concept: str, trace: List[str] = None) -> None:
        """
        Apply the Law of Ascension and Descent.
        Light/Happy concepts ascend (+Y, +W).
        Dark/Sad concepts descend (-Y, -W).
        """
        if not hasattr(self.memory, "causal_graph") or concept not in self.memory.causal_graph:
            return

        # 1. Determine Buoyancy based on Frequency (Proxy for Lightness)
        freq = self.vocabulary.get(concept, 0.5)
        
        # Center is Love (1.0)
        # > 0.7: Ascend (Light, Spirit)
        # < 0.4: Descend (Heavy, Matter)
        
        buoyancy = 0.0
        if freq >= 0.8: buoyancy = 0.1   # Strong Ascension
        elif freq >= 0.6: buoyancy = 0.05 # Weak Ascension
        elif freq <= 0.3: buoyancy = -0.1 # Strong Descent
        elif freq <= 0.5: buoyancy = -0.05 # Weak Descent
        
        if buoyancy == 0:
            return

        # 2. Update HyperQuaternion in Memory
        node = self.memory.causal_graph.nodes[concept]
        tensor_data = node.get("tensor", {})
        
        # Load Tensor (assuming dict)
        w = tensor_data.get("w", 1.0)
        y = tensor_data.get("y", 0.0)
        
        # Apply Force
        new_w = w + (buoyancy * 0.5) # Dimension changes slower
        new_y = y + buoyancy
        
        # Clamp/Boundaries
        # W: 0.0 (Point) to 4.0 (Hyper-God)
        new_w = max(0.1, min(4.0, new_w))
        # Y: -1.0 (Abyss/Body) to 1.0 (Heaven/Spirit)
        new_y = max(-1.0, min(1.0, new_y))
        
        # Save back
        tensor_data["w"] = new_w
        tensor_data["y"] = new_y
        node["tensor"] = tensor_data
        
        # Log to Trace if provided
        if trace is not None:
            direction = "Ascending ⇧" if buoyancy > 0 else "Descending ⇩"
            trace.append(f"⚖️ [Law] '{concept}' is {direction} (Y={new_y:.2f}, W={new_w:.2f})")

    def listen(self, text: str, t: float, visual_input: Dict[str, Any] = None) -> List[Tuple[str, Oscillator]]:
        """
        Convert user text AND visual input into a list of (concept, Oscillator) 'ripples'.
        """
        concepts = self._extract_concepts(text)
        
        # Process Visual Input
        if visual_input:
            # Brightness -> Light/Dark
            brightness = visual_input.get("brightness", 0)
            if brightness > 200: concepts.append("light")
            elif brightness < 50: concepts.append("shadow")
            
            # OCR Text -> Concepts
            ocr_text = visual_input.get("text", "")
            if ocr_text:
                concepts.extend(self._extract_concepts(ocr_text))

        ripples = []
        # We need a temporary trace for listen events, or we just log debug
        listen_trace = []
        
        for concept in concepts:
            # Register concept in memory (Birth of a Star)
            self.memory.add_concept(concept, concept_type="word")
            
            # Apply Law of Ascension
            self.apply_law_of_ascension(concept, trace=listen_trace)
            
            if concept in self.vocabulary:
                ripple = Oscillator(
                    amplitude=0.5,  # External ripples are strong
                    frequency=self.vocabulary[concept],
                    phase=(t * self.vocabulary[concept]) % (2 * math.pi) # Phase locked to time
                )
                ripples.append((concept, ripple))
        
        if listen_trace:
            logger.info(f"⚖️ Law of Ascension applied: {listen_trace}")
            # We might want to expose this to the user, but listen happens before speak.
            # We can store it in self.last_trace if we want, or append to a buffer.
            # For now, let's just log it.

        source_desc = f"Text='{text}'"
        if visual_input: source_desc += f", Vision={len(visual_input)} keys"
        logger.debug(f"Logos: Heard {source_desc} -> Created {len(ripples)} ripples.")
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

        num_to_select = min(len(concepts_in_sea), 5)  # Select up to 5 concepts
        thought_cloud = random.choices(concepts_in_sea, weights=weights, k=num_to_select)

        # Ensure the most probable concept is included
        most_probable_concept = max(probabilities, key=probabilities.get)
        if most_probable_concept not in thought_cloud and thought_cloud:
            thought_cloud[0] = most_probable_concept

        # 2b. Let the shared Hippocampus inject world/emergent concepts.
        try:
            graph_nodes = list(self.memory.causal_graph.nodes()) if hasattr(self.memory, "causal_graph") else []
        except Exception:
            graph_nodes = []

        if graph_nodes and thought_cloud:
            # Prefer to inject occasionally so emergent concepts surface in language.
            if random.random() < 0.6:
                injected = random.choice(graph_nodes)
                thought_cloud.append(injected)

        # 3. Construct the Response (Poetic Collapse)
        # Sort concepts by their resonance (amplitude * frequency)
        thought_cloud_sorted = sorted(
            thought_cloud,
            key=lambda c: (
                self.internal_sea[c].amplitude * self.internal_sea[c].frequency
                if c in self.internal_sea else 0
            ),
            reverse=True
        )

        # Form the response as a sentence.
        if not thought_cloud_sorted:
            return "..."
        
        response_parts = []
        for concept in thought_cloud_sorted[:3]:
            response_parts.append(concept)
        
        response = " ".join(response_parts)
        logger.info(f"💬 Response: {response}")
        
        return response

    def get_physical_action(self) -> Optional[Dict[str, Any]]:
        """
        Translate Quaternion State into a Physical Action.
        Mind -> Body connection.
        """
        if not self.consciousness_lens or not hasattr(self.consciousness_lens, 'state'):
            return None
            
        q = self.consciousness_lens.state.q
        
        # 1. Moral Axis (X): Gestures
        # Angels (+X) -> Nod, Smile (Positive)
        # Demons (-X) -> Shake, Frown (Negative)
        if q.x > 0.6:
            return {"type": "gesture", "name": "nod", "reason": "Agreement (Angel)"}
        elif q.x < -0.6:
            return {"type": "gesture", "name": "shake", "reason": "Disagreement (Demon)"}
            
        # 2. Trinity Axis (Y): Action
        # Spirit (+Y) -> Look Up
        # Body (-Y) -> Look Down
        if q.y > 0.8:
            return {"type": "look", "direction": "up", "reason": "Spirit (Ascension)"}
            
        return None

    def construct_fractal_thought(self, subject: str, target: str, action: str) -> str:
        """
        Constructs a thought using Fractal Grammar.
        """
        from Core.Life.grammar_physics import FractalSyntax
        syntax = FractalSyntax()
        return syntax.construct_sentence(subject, target, action)
    
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


# Re-export Oscillator for backward compatibility
from Core.Math.oscillator import Oscillator

__all__ = ['ResonanceEngine', 'Oscillator']

