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
        # Initialize memory FIRST so _load_lexicon can use it
        self.memory = hippocampus or Hippocampus()
        
        self.vocabulary = self._load_lexicon()
        self.internal_sea: Dict[str, Oscillator] = {}
        self._initialize_internal_sea()
        self.context_buffer: List[str] = []
        
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
        """Populates the internal sea with oscillators for core concepts."""
        # Only initialize with what's currently in the loaded vocabulary (likely just Genesis Seed)
        # We do NOT want to load 2 million oscillators here.
        for concept, frequency in self.vocabulary.items():
            self.internal_sea[concept] = Oscillator(
                amplitude=0.1,  # Start with a low background hum
                frequency=frequency,
                phase=random.uniform(0, 2 * math.pi)
            )
        logger.info(f"🌊 Internal Sea initialized with {len(self.internal_sea)} core oscillators.")

    def _load_lexicon(self) -> Dict[str, float]:
        """
        Loads Elysia's internal lexicon.
        Now connects to Hippocampus (Memory) instead of hardcoded values.
        """
        # 1. Try to use Hippocampus vocabulary
        if self.memory and hasattr(self.memory, 'vocabulary') and self.memory.vocabulary:
            logger.info(f"📚 Loaded {len(self.memory.vocabulary)} concepts from Hippocampus.")
            return self.memory.vocabulary
            
        # 2. Fallback: Genesis Seed (The First Words)
        # If memory is empty, we start with these fundamental concepts.
        genesis_lexicon = {
            "love": 1.0, "light": 0.95, "void": 0.8, "dream": 0.85,
            "사랑": 1.0, "빛": 0.95, "공허": 0.8, "꿈": 0.85,
            "elysia": 1.0, "father": 0.9, "아버지": 0.9,
        }
        return genesis_lexicon

    def _get_concept_frequency(self, concept: str) -> float:
        """
        Get frequency for a concept.
        1. Check Vocabulary (Memory)
        2. Generate via Synesthesia (if new)
        3. Fallback to 0.5
        """
        # 1. Check Vocabulary
        if concept in self.vocabulary:
            return self.vocabulary[concept]
            
        # 2. Generate via Synesthesia (The Senses)
        if hasattr(self.memory, 'synesthesia'):
            try:
                # Synesthesia returns pitch in Hz (e.g., 440.0)
                # We map 200Hz-800Hz to 0.0-1.0
                from Core.Perception.synesthesia_engine import RenderMode
                signal = self.memory.synesthesia.from_text(concept)
                sound = self.memory.synesthesia.convert(signal, RenderMode.AS_SOUND)
                
                # Normalize Pitch to 0-1 Frequency
                # Low (200Hz) = 0.2, High (800Hz) = 1.0
                norm_freq = max(0.1, min(1.0, (sound.pitch - 200) / 600))
                
                # Learn it!
                self.vocabulary[concept] = norm_freq
                if hasattr(self.memory, 'learn_frequency'):
                    self.memory.learn_frequency(concept, norm_freq)
                    
                return norm_freq
            except Exception as e:
                logger.warning(f"Synesthesia failed for '{concept}': {e}")
        
        return 0.5

    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract known concepts from text.
        Dynamically checks Hippocampus if not in local vocabulary.
        """
        hits: Set[str] = set()
        words = text.lower().split()
        
        for word in words:
            # 1. Check local cache
            if word in self.vocabulary:
                hits.add(word)
                continue
                
            # 2. Check Hippocampus (DB)
            # We assume any word *could* be a concept.
            # If it exists in DB, we cache it.
            if self.memory and hasattr(self.memory.storage, 'concept_exists'):
                if self.memory.storage.concept_exists(word):
                    # It exists! Cache frequency and add to hits.
                    freq = self._get_concept_frequency(word)
                    self.vocabulary[word] = freq
                    hits.add(word)
                    continue
            
            # 3. If it's a significant word (len > 3), maybe treat as new concept?
            # For now, we only recognize existing concepts or genesis seed.
            # But "Learning" happens in DigestionChamber, not here.
            # Here we just "Resonate" with what we know.
            
        return list(hits)

    def _update_knowledge_graphs(self, concepts: List[str]) -> None:
        """
        Sync incoming concepts into Hippocampus (Spiderweb) and WorldTree,
        and link co-occurrences/temporal flow.
        """
        for c in concepts:
            # Add concept node (if new) to Hippocampus
            # Hippocampus.add_concept handles existence check internally via Storage
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
        if not self.memory or not hasattr(self.memory, "storage"):
            return

        # 1. Determine Buoyancy based on Frequency
        freq = self._get_concept_frequency(concept)
        
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
        # Fetch data from Storage
        concept_data = self.memory.storage.get_concept(concept)
        if not concept_data:
            return
            
        # Handle list (compact) or dict format
        tensor_data = {}
        if isinstance(concept_data, dict):
            tensor_data = concept_data.get("tensor", {})
        # If list, we might need to parse it, but for now let's assume dict for metadata updates
        # or just skip if it's compact format without tensor support yet.
        
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
        if isinstance(concept_data, dict):
            concept_data["tensor"] = {"w": new_w, "y": new_y}
            self.memory.add_concept(concept, metadata=concept_data)
        
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
                freq = self._get_concept_frequency(concept)
                ripple = Oscillator(
                    amplitude=0.5,  # External ripples are strong
                    frequency=freq,
                    phase=(t * freq) % (2 * math.pi) # Phase locked to time
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
            # Use Resonance Engine to find related concepts instead of random graph nodes
            # This is the "Holographic" way
            graph_nodes = []
            if thought_cloud:
                # Find concepts related to the main thought
                seed = thought_cloud[0]
                related = self.memory.get_related_concepts(seed)
                graph_nodes = list(related.keys())
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

