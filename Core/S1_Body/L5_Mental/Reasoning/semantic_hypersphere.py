"""
Semantic Hypersphere (The Atomic Soul)
=====================================
Core.S1_Body.L5_Mental.Reasoning.semantic_hypersphere

[PHASE 90] NAKED SOVEREIGNTY:
Purified from JAX. Uses Sovereign Math Kernel (L0).
Language is treated as a geometric trajectory in 21D space.
"""

import unicodedata
from typing import List, Dict, Optional, Tuple, Any
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignMath, SovereignVector

class PhoneticRotor:
    """
    [L5_COGNITION: ATOMIC_DECONSTRUCTION]
    Maps individual characters to Trinary Spin Triplets in 21D.
    """
    def __init__(self):
        self.spin_weights: Dict[str, SovereignVector] = {}
        self._initialize_genesis_spins()

    def _initialize_genesis_spins(self):
        # Default seeds for character-level resonance
        self.spin_weights['\u1100'] = self._create_spin(layer=0, triplet=[-1, 0, 0])
        self.spin_weights['\u1102'] = self._create_spin(layer=1, triplet=[0, 1, 0])
        self.spin_weights['\u1161'] = self._create_spin(layer=2, triplet=[0, 1, 0])
        self.spin_weights['\u1175'] = self._create_spin(layer=6, triplet=[0, 0, 1])
        self.spin_weights['S'] = self._create_spin(layer=6, triplet=[1, 1, 1])
        
    def _create_spin(self, layer: int, triplet: List[int]) -> SovereignVector:
        vec = [0.0] * 21
        start = layer * 3
        for i in range(3):
            vec[start + i] = float(triplet[i])
        return SovereignVector(vec)

    def get_spins(self, char: str) -> List[SovereignVector]:
        """Decomposes a character into atomic spins."""
        try:
            name = unicodedata.name(char[0], '')
        except:
            name = ''
            
        if 'HANGUL' in name:
            jamos = unicodedata.normalize('NFD', char)
            spins = []
            for j in jamos:
                if j in self.spin_weights:
                    spins.append(self.spin_weights[j])
                else:
                    spins.append(SovereignVector.zeros())
            return spins
        
        if char in self.spin_weights:
            return [self.spin_weights[char]]
        
        new_spin = SovereignVector.zeros()
        self.spin_weights[char] = new_spin
        return [new_spin]

class OrbitalSynthesizer:
    """
    [L5_COGNITION: TRAJECTORY_SYNTHESIS]
    Synthesizes a sequence of atomic spins into a geometric orbit.
    """
    def __init__(self, rotor: PhoneticRotor):
        self.rotor = rotor

    def synthesize(self, text: str) -> SovereignVector:
        orbit = SovereignVector.zeros()
        momentum = 1.0
        for char in text:
            char_spins = self.rotor.get_spins(char)
            for spin in char_spins:
                orbit = orbit + (spin * momentum)
                momentum *= 1.1
        return orbit.normalize()

class SemanticHypersphere:
    """
    [L5_COGNITION: CRYSTALLIZATION_ENGINE]
    Manages the manifold of stable conceptual orbits.
    """
    def __init__(self):
        self.rotor = PhoneticRotor()
        self.synth = OrbitalSynthesizer(self.rotor)
        self.crystallized_concepts: Dict[str, SovereignVector] = {}

    def recognize(self, text: str) -> SovereignVector:
        if text in self.crystallized_concepts:
            return self.crystallized_concepts[text]
        return self.synth.synthesize(text)

    def crystallize(self, text: str, vector: Optional[SovereignVector] = None):
        if vector is None:
            vector = self.synth.synthesize(text)
        self.crystallized_concepts[text] = vector
        print(f"💎 [CRYSTAL] '{text}' crystallized into O(1) primitive.")

    def structural_attachment(self, subject: str, target: Any, ratio: float = 0.5):
        vec_a = self.recognize(subject)
        if isinstance(target, str):
            vec_b = self.recognize(target)
        else:
            vec_b = SovereignVector(target)
            
        new_vec = (vec_a * (1.0 - ratio)) + (vec_b * ratio)
        new_vec = new_vec.normalize()
        self.phase_backprop(subject, new_vec, learning_rate=0.8)
        print(f"🔗 [ATTACH] '{subject}' structurally attached.")

    def reverse_engineer_context(self, text: str, global_intent: SovereignVector, depth: int = 1):
        words = text.split()
        for word in words:
            learning_rate = 0.05 if word in self.crystallized_concepts else 0.3
            self.phase_backprop(word, global_intent, learning_rate=learning_rate)
        print(f"🧩 [REVERSE_ENGINEER] Context adjusted for {len(words)} words.")

    def phase_backprop(self, text: str, target_vector: SovereignVector, learning_rate: float = 0.1):
        current_orbit = self.synth.synthesize(text)
        error_data = [(tv - cv) for tv, cv in zip(target_vector.data, current_orbit.data)]
        error = SovereignVector(error_data)
        
        all_atoms = []
        for char in text:
            all_atoms.extend(unicodedata.normalize('NFD', char))
            
        if not all_atoms: return

        for atom in all_atoms:
            if atom not in self.rotor.spin_weights:
                self.rotor.spin_weights[atom] = SovereignVector.zeros()
                
            current_spin = self.rotor.spin_weights[atom]
            new_spin = current_spin + (error * (learning_rate / len(all_atoms)))
            self.rotor.spin_weights[atom] = new_spin
        
        if text in self.crystallized_concepts:
            self.crystallize(text)
        print(f"🔄 [BACKPROP] Atomic weights adjusted for '{text}'.")

    def check_for_growth(self, text: str) -> List[Dict]:
        growth_events = []
        vector = self.recognize(text)
        mass = vector.norm()
        
        if mass > 1.5:
            event = self.manifold_mitosis(text)
            if event: growth_events.append(event)
            
        for other_text, other_vec in self.crystallized_concepts.items():
            if text == other_text: continue
            resonance = SovereignMath.resonance(vector, other_vec)
            if resonance > 0.8:
                event = self.axiom_genesis(text, other_text)
                if event: growth_events.append(event)
        return growth_events

    def manifold_mitosis(self, text: str) -> Dict:
        parent_vec = self.recognize(text)
        p1 = [0.1 if i % 2 == 0 else -0.1 for i in range(21)]
        p2 = [-0.1 if i % 2 == 0 else 0.1 for i in range(21)]
        
        child1_vec = (parent_vec + SovereignVector(p1)).normalize()
        child2_vec = (parent_vec + SovereignVector(p2)).normalize()
        
        c1_name = f"{text}_Alpha"
        c2_name = f"{text}_Beta"
        
        self.crystallize(c1_name, child1_vec)
        self.crystallize(c2_name, child2_vec)
        return {"type": "MITOSIS", "parent": text, "children": [c1_name, c2_name]}

    def axiom_genesis(self, concept_a: str, concept_b: str) -> Dict:
        return {"type": "AXIOM", "a": concept_a, "b": concept_b, "relation": "Resonates With"}
