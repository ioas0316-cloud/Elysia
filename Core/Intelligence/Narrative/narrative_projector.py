"""
Narrative Projector (서사 투영기)
=================================
Core.Intelligence.Narrative.narrative_projector

"The Novel is a shadow of the 4D Truth."
"소설은 4차원 진실의 그림자다."

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

from Core.Intelligence.narrative_weaver import THE_BARD
from Core.Foundation.Wave.wave_dna import WaveDNA

class NarrativeProjector:
    def __init__(self):
        self.output_log = "c:\\Elysia\\data\\elysian_chronicle.txt"
        os.makedirs(os.path.dirname(self.output_log), exist_ok=True)
        
        # Mapping WaveDNA sectors to Narrative motifs
        self.motifs = {
            "physical": ["단단한 현실", "육체적 갈망", "대지의 울림", "거친 숨결"],
            "phenomenal": ["일렁이는 감각", "색채의 향연", "아득한 향기", "찰나의 질감"],
            "mental": ["차가운 논리", "정교한 계산", "유리 같은 사유", "구조적 의심"],
            "spiritual": ["신성한 불꽃", "영혼의 고동", "초월적 빛", "심연의 안식"],
            "causal": ["인과의 사슬", "예정된 비극", "운명의 회전", "필연적 붕괴"]
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
            
        motif = random.choice(self.motifs.get(dominant_sector, ["평범한 일상"]))
        
        # 2. Extract Base Prose from the Bard
        base_prose = THE_BARD.elaborate_ko(actor_name, event_type, motif, era_name)
        
        # 3. Apply "Psionic Distortion" (Styling based on intensities)
        # Higher spiritual/phenomenal makes prose more abstract
        intensity = field_state.phenomenal + field_state.spiritual
        if intensity > 1.2:
            base_prose = f"주변의 현실이 흐릿해지며... {base_prose}"
        elif field_state.physical > 0.8:
            base_prose = f"뼈저린 고통 속에서, {base_prose}"
            
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
