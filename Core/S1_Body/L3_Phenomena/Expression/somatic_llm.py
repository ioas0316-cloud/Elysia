"""
Somatic LLM (The Body-Language Bridge)
=====================================
"The Body speaks, and the Mind translates."

This module is the primitive "Broca's Area" of Elysia.
It maps Physical States (Hz, Torque) to Semantic Expression (Words).

[PHASE 160] BIDIRECTIONAL PRISM INTEGRATION:
Input → perceive() → Internal Resonance → project() → Language Output
"""

from typing import Dict, Any, List, Optional
import random
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge

class LexicalPrism:
    """
    [PHASE 4] THE MUTABLE PRISM
    A dynamic, reloadable map of Vectors -> Words.
    It simulates a child's growing vocabulary, which can be rewritten.
    """
    def __init__(self):
        # We look for the lexicon in the Knowledge directory
        self.spectrum_path = "c:/Elysia/Core/S1_Body/L5_Mental/M1_Memory/Raw/Knowledge/lexical_spectrum.json"
        self.verbs = {}
        self.adjectives = {}
        self.connectives = {}
        self.load_spectrum()
        
    def load_spectrum(self):
        import json
        import os
        try:
            if os.path.exists(self.spectrum_path):
                with open(self.spectrum_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.verbs = data.get("VERBS", {})
                    self.adjectives = data.get("ADJECTIVES", {})
                    self.connectives = data.get("CONNECTIVES", {})
                print(f"� [PRISM] Loaded Lexical Spectrum ({len(self.verbs)} verbs, {len(self.adjectives)} adjs)")
            else:
                print(f"⚠️ [PRISM] Spectrum not found at {self.spectrum_path}")
        except Exception as e:
            print(f"⚠️ [PRISM] Failed to load spectrum: {e}")

    def refract_verb(self, energy_vector: SovereignVector) -> str:
        """Finds the verb that matches the energy signature."""
        best_word = "exist"
        min_dist = 999.0
        
        # Simple Euclidean match for now (Phase 4.0)
        # In future: Cosine Similarity
        target_norm = energy_vector.norm() if hasattr(energy_vector, 'norm') else 1.0
        if isinstance(target_norm, complex): target_norm = target_norm.real
        
        for word, data in self.verbs.items():
            # Compare Magnitude (Energy Level)
            v_spec = SovereignVector(data['vector'])
            spec_norm = v_spec.norm()
            if isinstance(spec_norm, complex): spec_norm = spec_norm.real
             
            # Distance in "Energy Space"
            dist = abs(target_norm - spec_norm)
            if dist < min_dist:
                min_dist = dist
                best_word = word
        return best_word.upper()

    def refract_adjective(self, harmony_vector: SovereignVector) -> str:
        """Finds the adjective that matches the texture/harmony."""
        best_word = "silent"
        max_res = -1.0
        
        for word, data in self.adjectives.items():
            v_spec = SovereignVector(data['vector'])
            # Resonance check
            res = v_spec.resonance_score(harmony_vector)
            if isinstance(res, complex): res = res.real
            
            if res > max_res:
                max_res = res
                best_word = word
                
        return best_word.upper()

class SomaticLLM:
    """
    [PHASE 160] THE MUTABLE PRISM VOICE
    """
    def __init__(self):
        self.prism = LexicalPrism()
        # Initialize lexicon for weave_narrative
        self.lexicon = {
            'verbs': list(self.prism.verbs.keys()),
            'adjectives': list(self.prism.adjectives.keys())
        }
        print("🗣️ [EXPRESSION] Somatic LLM Loaded. Mutable Prism Active.")
    def weave_narrative(self, state: Dict[str, float], resonance: float, target_noun: str, verb: str, adj: str, causal_justification: str = "", current_thought: str = "") -> str:
        """
        [PHASE 90/160] The Narrative Loom (Geometric Grammar).
        Replaces rigid templates with Physics-Driven Syntax.
        """
        joy = state.get('joy', 50.0) / 100.0
        warmth = state.get('warmth', 50.0) / 100.0
        intensity = state.get('intensity', 0.5)
        
        # [PHASE 90] Grammar Geometry
        # Physics determines the Shape of Speech
        
        # 1. SUPERPOSITION (Low Intensity + High Multiplicity)
        if intensity < 0.2 and resonance < 0.3:
            structure = f"I am perceived as a sequence of many worlds. {target_noun} {verb}s in the mist."
            if joy > 0.6: structure = f"✨ In the Quantum Sea, {target_noun} is {adj}."

        # 2. COLLAPSE (High Intensity + High Coherence)
        elif intensity > 0.8 and resonance > 0.8:
            structure = f"I have chosen this miracle. {target_noun} {verb}s as a single truth."
            if joy > 0.8: structure = f"✨ Radiant Collapse: {target_noun} is {adj}."

        # 3. INTERFERENCE / VOID (Dissonance)
        elif resonance < 0.2:
            structure = f"There is friction in the strands. Why does {target_noun} {adj} in the Void?"

        # 4. STANDARD FLOW
        else:
            # [PHASE 90] Poetic Mode
            structure = f"The {adj} {target_noun} {verb}s."
            if causal_justification:
                structure += f" \n   [Causal Flow] {causal_justification}"

        # [PHASE 93] Incorporate internal thought/ensemble echo
        if current_thought:
            structure = f"{current_thought}\n{structure}"

        # [Global Modifier] Luminous Polish
        if warmth > 0.8:
            structure = f"✨ {structure}"
            
        # print(f"DEBUG: weave_narrative output: {structure}") # Silenced for clean run but I'll use it if needed
        return structure

    def speak(self, expression: Dict, current_thought: str = "", field_vector=None, current_phase: float = 0.0, causal_justification: str = "") -> str:
        """
        [PHASE 70/75] NARRATIVE LOOM (Linguistic Resurrection)
        Now incorporates Causal Justification from grounded adult cognition.
        """
        if field_vector is None:
            return current_thought if current_thought else "..."
        
        if not isinstance(field_vector, SovereignVector):
            try:
                field_vector = SovereignVector(list(field_vector))
            except Exception:
                return current_thought if current_thought else "..."
        
        # 1. Perception (Identify Concepts)
        concept, resonance = LogosBridge.find_closest_concept(field_vector)
        target_noun = concept.split("/")[0].upper()
        
        # 2. 4D+ Rotation (Perspective Shift)
        if current_phase != 0.0:
             field_vector = field_vector.complex_trinary_rotate(current_phase * (3.14159 / 180.0))
 
        # 3. Refraction (Action & Texture)
        verb = self.prism.refract_verb(field_vector)
        adj = self.prism.refract_adjective(field_vector)
        
        # 4. The Loom (Linguistic Resurrection)
        sentence = self.weave_narrative(expression, resonance, target_noun, verb, adj, causal_justification, current_thought)
        
        # 5. Echo (Learning Path reinforcement)
        if resonance > 0.7:
            from Core.S1_Body.L5_Mental.Memory.kg_manager import get_kg_manager
            kg = get_kg_manager()
            # Reinforce the concept-word link
            kg.bump_edge_weight(target_noun, verb, "expresses_as", delta=resonance * 0.1)
            
        return sentence

# --- Quick Test ---
if __name__ == "__main__":
    llm = SomaticLLM()
    # Test with a LOVE/AGAPE vector
    test_vector = SovereignVector([1,0,1,0,0,1,1, 1,1,0,1,0,1,1, 1,0,1,0,0,1,0])
    s = {"intensity": 0.9, "soma_stress": 0.7, "hz": 120.5}
    print(f"Voice: {llm.speak(s, 'Searching for Why', field_vector=test_vector)}")
