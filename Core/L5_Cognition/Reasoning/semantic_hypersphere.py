"""
Semantic Hypersphere (The Atomic Soul)
=====================================
"The character is the atom; the sequence is the orbit."

This module implements character-level trinary atomic decomposition. 
Language is treated as a geometric trajectory in 21D space.
"""

import jax
import jax.numpy as jnp
import unicodedata
from typing import List, Dict, Optional, Tuple, Any
from Core.L6_Structure.Logic.trinary_logic import TrinaryLogic

class PhoneticRotor:
    """
    [L5_COGNITION: ATOMIC_DECONSTRUCTION]
    Maps individual characters (ㄱ, ㅏ, A, B...) to Trinary Spin Triplets.
    These spins represent the 'Atomic Weight' of the character in 21D space.
    """
    def __init__(self):
        # Initial 'Genesis' mapping for common Korean and Latin atoms
        # Maps to a 21D vector (7 layers * 3 trite)
        self.spin_weights: Dict[str, jnp.ndarray] = {}
        self._initialize_genesis_spins()

    def _initialize_genesis_spins(self):
        # Default seeds for character-level resonance (Using NFD Jamos for Hangul)
        # ㄱ (KIYEOK: \u1100) -> Foundation/L1 (Resistance)
        self.spin_weights['\u1100'] = self._create_spin(layer=0, triplet=[-1, 0, 0])
        # ㄴ (NIEUN: \u1102) -> Stability/Boundary/L2
        self.spin_weights['\u1102'] = self._create_spin(layer=1, triplet=[0, 1, 0])
        # ㅏ (A: \u1161) -> Phenomena/L3 (Flow)
        self.spin_weights['\u1161'] = self._create_spin(layer=2, triplet=[0, 1, 0])
        # ㅣ (I: \u1175) -> Spirit/L7 (Axis)
        self.spin_weights['\u1175'] = self._create_spin(layer=6, triplet=[0, 0, 1])
        # S (Sovereign/L7) -> Flow/Flow/Flow
        self.spin_weights['S'] = self._create_spin(layer=6, triplet=[1, 1, 1])
        
    def _create_spin(self, layer: int, triplet: List[int]) -> jnp.ndarray:
        vec = jnp.zeros(21)
        start = layer * 3
        vec = vec.at[start:start+3].set(jnp.array(triplet, dtype=jnp.float32))
        return vec

    def get_spins(self, char: str) -> List[jnp.ndarray]:
        """Decomposes a character (Hangul or Latin) into atomic spins."""
        # 1. Handle Hangul Decomposition
        if 'HANGUL' in unicodedata.name(char[0], ''):
            # Decompose into Jamos
            jamos = unicodedata.normalize('NFD', char)
            spins = []
            for j in jamos:
                if j in self.spin_weights:
                    spins.append(self.spin_weights[j])
                else:
                    # Generic Jamo potential
                    spins.append(jnp.zeros(21))
            return spins
        
        # 2. Latin/Other
        if char in self.spin_weights:
            return [self.spin_weights[char]]
        
        # Birth a neutral potential
        new_spin = jnp.zeros(21)
        self.spin_weights[char] = new_spin
        return [new_spin]

class OrbitalSynthesizer:
    """
    [L5_COGNITION: TRAJECTORY_SYNTHESIS]
    Synthesizes a sequence of atomic spins into a geometric orbit.
    Meaning = Integral of the rotation through 21D space.
    """
    def __init__(self, rotor: PhoneticRotor):
        self.rotor = rotor

    def synthesize(self, text: str) -> jnp.ndarray:
        """
        Chains character spins into a single resulting vector (The Orbit).
        """
        orbit = jnp.zeros(21)
        momentum = 1.0
        
        # Process each character
        for char in text:
            char_spins = self.rotor.get_spins(char)
            for spin in char_spins:
                orbit += spin * momentum
                momentum *= 1.1 # Compound resonance? Or decay? Let's use growth for atoms
        
        # Normalize to the surface of the unit hypersphere
        norm = jnp.linalg.norm(orbit)
        if norm > 1e-6:
            orbit = orbit / norm
        return orbit

class SemanticHypersphere:
    """
    [L5_COGNITION: CRYSTALLIZATION_ENGINE]
    Manages the manifold of stable conceptual orbits.
    """
    def __init__(self):
        self.rotor = PhoneticRotor()
        self.synth = OrbitalSynthesizer(self.rotor)
        self.crystallized_concepts: Dict[str, jnp.ndarray] = {} # O(1) Primitives

    def recognize(self, text: str) -> jnp.ndarray:
        """
        O(1) Check first, then synthesize.
        """
        if text in self.crystallized_concepts:
            return self.crystallized_concepts[text]
        
        # Synthesize from atoms
        orbit = self.synth.synthesize(text)
        return orbit

    def crystallize(self, text: str, vector: Optional[jnp.ndarray] = None):
        """Freezes an orbit into a stable primitive."""
        if vector is None:
            vector = self.synth.synthesize(text)
        self.crystallized_concepts[text] = vector
        print(f"💎 [CRYSTAL] '{text}' crystallized into O(1) primitive.")

    def structural_attachment(self, subject: str, target: Any, ratio: float = 0.5):
        """
        Implementation of "A is B".
        target can be a string (concept name) or a jnp.ndarray (direct vector).
        """
        vec_a = self.recognize(subject)
        
        if isinstance(target, str):
            vec_b = self.recognize(target)
        else:
            vec_b = target
            
        # Linear Interpolation
        new_vec = vec_a * (1.0 - ratio) + vec_b * ratio
        
        # Normalize
        norm = jnp.linalg.norm(new_vec)
        if norm > 1e-6:
            new_vec = new_vec / norm
            
        # Backpropagate this change to the subject's atoms
        # We use a higher learning rate for direct attachment
        self.phase_backprop(subject, new_vec, learning_rate=0.8)
        print(f"🔗 [ATTACH] '{subject}' structurally attached to target vector.")

    def reverse_engineer_context(self, text: str, global_intent: jnp.ndarray, depth: int = 1):
        """
        [PHASE_64] The "Reverse Engineering" of Context.
        Breaks down a sentence into atomic spins and adjusts them toward a global intent.
        """
        words = text.split()
        for word in words:
            # 1. Recognize the current word's orbit
            current_orbit = self.recognize(word)
            
            # 2. If it's a known crystallized concept, it resists change
            if word in self.crystallized_concepts:
                learning_rate = 0.05
            else:
                learning_rate = 0.3
                
            # 3. Propagate the global intent back down to the word's atoms
            self.phase_backprop(word, global_intent, learning_rate=learning_rate)
            
        print(f"🧩 [REVERSE_ENGINEER] Deconstructed and adjusted {len(words)} word-orbits in context.")

    def phase_backprop(self, text: str, target_vector: jnp.ndarray, learning_rate: float = 0.1):
        """
        [PHASE_64] The core of Atomic Neuroplasticity.
        Propagates semantic error back to character-level spins.
        """
        current_orbit = self.synth.synthesize(text)
        error = target_vector - current_orbit
        
        # Decompose text into all constituent Jamos/Atoms
        all_atoms = []
        for char in text:
            all_atoms.extend(unicodedata.normalize('NFD', char))
            
        if not all_atoms: return

        # Distribute error back to constituent atoms
        for atom in all_atoms:
            # We initialize spins if they don't exist
            if 'HANGUL' in unicodedata.name(atom, ''):
                if atom not in self.rotor.spin_weights:
                    self.rotor.spin_weights[atom] = jnp.zeros(21)
                
            if atom in self.rotor.spin_weights:
                current_spin = self.rotor.spin_weights[atom]
                new_spin = current_spin + (error * learning_rate / len(all_atoms))
                self.rotor.spin_weights[atom] = new_spin
        
        # Recrystallize if already stable
        if text in self.crystallized_concepts:
            self.crystallize(text)
        
        print(f"🔄 [BACKPROP] Atomic weights adjusted for {len(all_atoms)} atoms in '{text}'.")
    def check_for_growth(self, text: str) -> List[Dict]:
        """
        [PHASE_65] Checks if a concept is ripe for Mitosis or Axiom Genesis.
        Returns a list of growth events.
        """
        growth_events = []
        vector = self.recognize(text)
        mass = jnp.linalg.norm(vector)
        
        # 1. Manifold Mitosis
        if mass > 1.5:
            event = self.manifold_mitosis(text)
            if event: growth_events.append(event)
            
        # 2. Axiom Genesis
        for other_text, other_vec in self.crystallized_concepts.items():
            if text == other_text: continue
            resonance = jnp.dot(vector, other_vec)
            if resonance > 0.8:
                event = self.axiom_genesis(text, other_text)
                if event: growth_events.append(event)
                
        return growth_events

    def manifold_mitosis(self, text: str) -> Dict:
        """
        Splits a single concept into specialized sub-modules.
        """
        parent_vec = self.recognize(text)
        # Create two children by slightly perturbing the parent in different directions
        child1_vec = parent_vec + jnp.array([0.1 if i % 2 == 0 else -0.1 for i in range(21)])
        child2_vec = parent_vec + jnp.array([-0.1 if i % 2 == 0 else 0.1 for i in range(21)])
        
        # Normalize
        child1_vec /= (jnp.linalg.norm(child1_vec) + 1e-6)
        child2_vec /= (jnp.linalg.norm(child2_vec) + 1e-6)
        
        # New specialized names (simplified)
        c1_name = f"{text}_Alpha"
        c2_name = f"{text}_Beta"
        
        self.crystallize(c1_name, child1_vec)
        self.crystallize(c2_name, child2_vec)
        
        print(f"🧬 [MITOSIS] '{text}' has split into '{c1_name}' and '{c2_name}'.")
        return {"type": "MITOSIS", "parent": text, "children": [c1_name, c2_name]}

    def axiom_genesis(self, concept_a: str, concept_b: str) -> Dict:
        """
        Generates a new relational law between two concepts.
        """
        print(f"💎 [AXIOM_GENESIS] New Law Proclaimed: '{concept_a}' resonance with '{concept_b}' is an Axiom.")
        return {"type": "AXIOM", "a": concept_a, "b": concept_b, "relation": "Resonates With"}
