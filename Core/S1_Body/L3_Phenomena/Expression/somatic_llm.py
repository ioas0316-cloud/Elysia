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
        print("🗣️ [EXPRESSION] Somatic LLM Loaded. Mutable Prism Active.")

    def speak(self, expression: Dict, current_thought: str = "", field_vector=None, current_phase: float = 0.0) -> str:
        """
        [PHASE 4/18: 4D REFRACTION]
        Constructs a sentence atom-by-atom using the Lexical Prism and Rotor Phase.
        """
        # If no field vector, fall back to legacy behavior (or calculate from thought)
        if field_vector is None:
            return current_thought if current_thought else "..."
        
        # Convert to SovereignVector if needed
        if not isinstance(field_vector, SovereignVector):
            try:
                field_vector = SovereignVector(list(field_vector))
            except Exception:
                return current_thought if current_thought else "..."
        
        # 1. Identify the Target (Noun)
        # [PHASE 18] Passing phase to ensure the search is hyperspherically rotated
        concept, resonance = LogosBridge.find_closest_concept(field_vector)
        target_noun = concept.split("/")[0].upper()
        
        # [PHASE 18] Use current_phase to modulate the prism refraction
        # A simple rotation of the field vector before refraction to simulate "multi-angle" view
        if current_phase != 0.0:
             field_vector = field_vector.complex_trinary_rotate(current_phase * (3.14159 / 180.0))

        # 2. Refract Verb (Action/Energy)
        verb = self.prism.refract_verb(field_vector)
        
        # 3. Refract Adjective (Texture/Harmony)
        adj = self.prism.refract_adjective(field_vector)
        
        # 4. Construct Emergent Sentence
        connective = "towards" 
        sentence = f"I [{verb}] {connective} the [{adj}] [{target_noun}]."
        
        if resonance > 0.8:
            sentence += " (LIGHT!)"
            
        return sentence

# --- Quick Test ---
if __name__ == "__main__":
    llm = SomaticLLM()
    # Test with a LOVE/AGAPE vector
    test_vector = SovereignVector([1,0,1,0,0,1,1, 1,1,0,1,0,1,1, 1,0,1,0,0,1,0])
    s = {"intensity": 0.9, "soma_stress": 0.7, "hz": 120.5}
    print(f"Voice: {llm.speak(s, 'Searching for Why', field_vector=test_vector)}")
