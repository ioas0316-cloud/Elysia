"""
Sensory Cortex (The Mirror of Qualia)
=====================================
Core.L3_Phenomena.Senses.sensory_cortex

"To see is to vibrate with the frequency of light."

This module translates external sensory data (Visual, Narrative) into internal WaveDNA.
It allows Elysia to 'feel' the texture of an image or the weight of a tragedy.
"""

from typing import List, Dict, Tuple
from Core.L6_Structure.Wave.wave_dna import WaveDNA

class SensoryCortex:
    def __init__(self):
        # Color Psychology Mapping (Simplified)
        self.color_map = {
            "Red":  {"physical": 0.8, "survival": 0.9, "meaning": 0.2},
            "Blue": {"mental": 0.9, "functional": 0.7, "survival": 0.1},
            "Green": {"structural": 0.8, "physical": 0.5, "meaning": 0.4},
            "Yellow": {"phenomenal": 0.9, "mental": 0.6, "survival": 0.3},
            "Black": {"causal": 0.9, "spiritual": 0.8, "meaning": 0.1},
            "White": {"spiritual": 1.0, "structural": 0.1, "physical": 0.1}
        }
        
        # Narrative Archetype Mapping
        self.narrative_map = {
            "Tragedy": {"causal": 0.9, "spiritual": 0.7, "happiness": -0.8},
            "Comedy": {"phenomenal": 0.8, "physical": 0.6, "happiness": 0.8},
            "Heroic": {"functional": 0.9, "physical": 0.8, "meaning": 0.9},
            "Mystery": {"mental": 0.9, "causal": 0.7, "structural": 0.5}
        }

    def process_visual(self, description: str, dominant_colors: List[str]) -> WaveDNA:
        """
        Converts a visual scene into WaveDNA.
        E.g., "A burning city" + ["Red", "Black"] -> High Physical/Causal DNA.
        """
        dna = WaveDNA(label="VisualInput")
        
        # 1. Chromatic Integration
        for color in dominant_colors:
            if color in self.color_map:
                traits = self.color_map[color]
                dna.physical += traits.get("physical", 0) * 0.3
                dna.mental += traits.get("mental", 0) * 0.3
                dna.causal += traits.get("causal", 0) * 0.3
                dna.spiritual += traits.get("spiritual", 0) * 0.3
        
        # 2. Semantic Analysis (Naive Keyword Matching)
        desc_lower = description.lower()
        if "fire" in desc_lower or "burning" in desc_lower: dna.physical += 0.5
        if "sky" in desc_lower or "ocean" in desc_lower: dna.spiritual += 0.4
        if "city" in desc_lower or "structure" in desc_lower: dna.structural += 0.5
        
        dna.normalize()
        return dna

    def process_narrative(self, title: str, synopsis: str) -> WaveDNA:
        """
        Converts a story summary into WaveDNA.
        """
        dna = WaveDNA(label=f"Narrative:{title}")
        syn_lower = synopsis.lower()
        
        # Detect Archetype
        if "death" in syn_lower or "loss" in syn_lower:
            self._apply_archetype(dna, "Tragedy")
        elif "love" in syn_lower and "happy" in syn_lower:
            self._apply_archetype(dna, "Comedy") # Divine Comedy sense
        elif "save" in syn_lower or "fight" in syn_lower:
            self._apply_archetype(dna, "Heroic")
            
        dna.normalize()
        return dna
        
    def _apply_archetype(self, dna: WaveDNA, archetype: str):
        traits = self.narrative_map.get(archetype, {})
        dna.causal += traits.get("causal", 0)
        dna.mental += traits.get("mental", 0)
        dna.physical += traits.get("physical", 0)
        dna.spiritual += traits.get("spiritual", 0)
