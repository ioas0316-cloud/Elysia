"""
Narrative Projector (주권적 자아)
=================================
Core.L5_Mental.Intelligence.Narrative.narrative_projector

"The Novel is a shadow of the 4D Truth."
"    4           ."

Features:
- Psionic Projection: 4D Field State -> 2D Prose.
- Spin-to-Emotion Mapping: Rotor RPM affects narrative intensity.
- Resonance Weaving: Finding metaphors for current Hypersphere coordinates.
"""

import os
import sys
import random
from typing import Dict, List, Any

# Ensure Core is visible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.L5_Mental.Intelligence.narrative_weaver import THE_BARD
from Core.L6_Structure.Wave.wave_dna import WaveDNA

class NarrativeProjector:
    def __init__(self):
        self.output_log = "c:\\Elysia\\data\\elysian_chronicle.txt"
        os.makedirs(os.path.dirname(self.output_log), exist_ok=True)
        
        # Mapping WaveDNA sectors to Narrative motifs
        self.motifs = {
            "physical": ["      ", "      ", "      ", "     "],
            "phenomenal": ["       ", "      ", "      ", "      "],
            "mental": ["      ", "      ", "        ", "      "],
            "spiritual": ["      ", "      ", "     ", "      "],
            "causal": ["      ", "      ", "      ", "      "]
        }

    def project_event(self, actor_name: str, event_type: str, field_state: WaveDNA, era_name: str):
        """
        Transmutes a 4D event into 2D prose.
        """
        # 1. Identify dominant motif from WaveDNA
        if hasattr(field_state, 'get_dominant_sector'):
            dominant_sector = field_state.get_dominant_sector()
        else:
            # Manual check if method missing
            sectors = {
                "physical": field_state.physical,
                "phenomenal": field_state.phenomenal,
                "mental": field_state.mental,
                "spiritual": field_state.spiritual,
                "causal": field_state.causal
            }
            dominant_sector = max(sectors, key=sectors.get)
            
        motif = random.choice(self.motifs.get(dominant_sector, ["      "]))
        
        # 2. Extract Base Prose from the Bard
        base_prose = THE_BARD.elaborate_ko(actor_name, event_type, motif, era_name)
        
        # 3. Apply "Psionic Distortion" (Styling based on intensities)
        # Higher spiritual/phenomenal makes prose more abstract
        intensity = field_state.phenomenal + field_state.spiritual
        if intensity > 1.2:
            base_prose = f"             ... {base_prose}"
        elif field_state.physical > 0.8:
            base_prose = f"          , {base_prose}"
            
        # 4. Final Formatting
        timestamp = "[4D Vector Sync]"
        final_line = f"{timestamp} {actor_name} | {base_prose}\n"
        
        # Append to Chronicle
        with open(self.output_log, "a", encoding="utf-8") as f:
            f.write(final_line)
            
        return final_line

    def clear(self):
        if os.path.exists(self.output_log):
            os.remove(self.output_log)

# Singleton
THE_PROJECTOR = NarrativeProjector()
