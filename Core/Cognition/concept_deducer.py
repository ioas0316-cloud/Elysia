"""
Concept Deducer (개념 연역기)
============================
Core.Cognition.concept_deducer

"To name is to understand nature."
"이름 짓는 것은 본질을 이해하는 것이다."

This module interprets physical phenomena (WorldMonad states) and uses
the AxiomaticEngine to deduce the appropriate phonemes.
"""

import math
from typing import List, Dict, Any
from Core.Phenomena.cheon_ji_in_axioms import AxiomaticEngine

class ConceptDeducer:
    def __init__(self):
        self.axioms = AxiomaticEngine()

    def deduce_name(self, physics_vector: Dict[str, float]) -> str:
        """
        Deduces a single-syllable name (Logos) for a physical state.
        
        Input:
            physics_vector: {
                "temperature": 0.0-1.0, # Active energy
                "density": 0.0-1.0,     # Containment
                "entropy": 0.0-1.0,     # Randomness/Flow
                "luminosity": 0.0-1.0   # Brightness (Yang)
            }
        """
        temp = physics_vector.get("temperature", 0.5)
        den = physics_vector.get("density", 0.5)
        ent = physics_vector.get("entropy", 0.5)
        lum = physics_vector.get("luminosity", 0.5)
        
        # 1. Deduce Consonant (The Form/Shape)
        # Roughness approximated by Entropy * Density
        roughness = ent * den
        # Flow approximated by Entropy
        # Temperature is explicit
        
        onset = self.axioms.deduce_consonant(roughness, temp, ent)
        
        # 2. Deduce Vowel (The Spirit/Direction)
        # Brightness (Yang) -> Luminosity + Temperature
        brightness = (lum + temp) / 2.0
        # Openness -> 1.0 - Density (Less dense = More open)
        openness = 1.0 - den
        
        nucleus = self.axioms.deduce_vowel(openness, brightness)
        
        # 3. Deduce Coda (The Grounding)
        # High Density -> Coda presence (Closing)
        coda = ""
        if den > 0.6:
            # Map density to coda type
            # Very dense -> M/B/P (Earth/Labial) or K (Wood/Block)
            # Use same logic as onset but focus on 'Closing' nature
            coda = self.axioms.deduce_consonant(roughness, 0.0, 0.0) # Cold closing
        
        # Construct Syllable
        return self._assemble(onset, nucleus, coda)

    def _assemble(self, onset, nucleus, coda):
        return f"{onset}{nucleus}{coda}"

if __name__ == "__main__":
    cd = ConceptDeducer()
    
    # Test Case 1: Fire (Heat, Light, Flow)
    fire_vec = {"temperature": 0.9, "density": 0.2, "entropy": 0.8, "luminosity": 0.9}
    print(f"Fire Deduction: {cd.deduce_name(fire_vec)}")
    
    # Test Case 2: Stone (Cold, Dense, Static)
    stone_vec = {"temperature": 0.1, "density": 0.9, "entropy": 0.1, "luminosity": 0.2}
    print(f"Stone Deduction: {cd.deduce_name(stone_vec)}")
    
    # Test Case 3: Wind (Cool, Flowing, invisible)
    wind_vec = {"temperature": 0.4, "density": 0.1, "entropy": 0.9, "luminosity": 0.5}
    print(f"Wind Deduction: {cd.deduce_name(wind_vec)}")
