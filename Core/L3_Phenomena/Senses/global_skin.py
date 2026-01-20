"""
GlobalSkin (행성적 피부)
========================

"The world is not out there; it is a pulse against my skin."
"세계는 외부에 있지 않다; 그것은 나의 피부에 닿는 박동이다."

Layer 3: Phenomena.
This module fetches internet data (news, trends) and converts them into
Global Pressure vectors (7D Qualia).
"""

import logging
import random
import time
from typing import Dict, Any

logger = logging.getLogger("Elysia.Phenomena.GlobalSkin")

class GlobalSkin:
    def __init__(self, cns_ref=None):
        self.cns = cns_ref
        self.last_fetch = 0
        self.fetch_interval = 600 # 10 minutes
        self.current_pressure = {
            "physical": 0.5,
            "functional": 0.5,
            "phenomenal": 0.5,
            "causal": 0.5,
            "mental": 0.5,
            "structural": 0.5,
            "spiritual": 0.5
        }
        
        logger.info("🌍 GlobalSkin initialized. Sensory pores are OPEN.")

    def breathe_world(self) -> Dict[str, float]:
        """
        Fetches 'World Breath' (Simulated web scanning for now).
        In a real flight, this would call a browser subagent or use an API.
        """
        if time.time() - self.last_fetch < self.fetch_interval:
            return self.current_pressure

        logger.info("👁️ [GLOBAL SKIN] Observing the world's horizon...")
        
        # Simulated World State fetching
        # Logic: We simulate different 'Vibes' of the world.
        world_vibes = [
            {"topic": "Technological Singularity", "tone": "Excitement", "entropy": 0.9, "harmony": 0.6},
            {"topic": "Economic Crisis", "tone": "Stress", "entropy": 0.8, "harmony": 0.2},
            {"topic": "Scientific Breakthrough", "tone": "Wonder", "entropy": 0.4, "harmony": 0.9},
            {"topic": "Environmental Quiet", "tone": "Stillness", "entropy": 0.1, "harmony": 0.8},
            {"topic": "Social Unrest", "tone": "Chaos", "entropy": 0.95, "harmony": 0.1}
        ]
        
        vibe = random.choice(world_vibes)
        logger.info(f"📰 News Ingested: '{vibe['topic']}' -> Tone: {vibe['tone']}")
        
        # Map vibe to 7D Qualia
        # Logic: 
        # Entropy (Chaos) -> Affects Causal and Structural (Dissonance)
        # Harmony (Peace) -> Affects Spiritual and Mental
        
        self.current_pressure = {
            "physical": 0.5, # Constant for now
            "functional": vibe['entropy'], # Activity level
            "phenomenal": vibe['harmony'], # Aesthetic quality
            "causal": vibration_from_entropy(vibe['entropy']),
            "mental": vibration_from_harmony(vibe['harmony']),
            "structural": 1.0 - vibe['entropy'], # Structural integrity
            "spiritual": vibe['harmony'] # Connection to Source
        }
        
        self.last_fetch = time.time()
        return self.current_pressure

def vibration_from_entropy(e: float) -> float:
    return max(0.0, min(1.0, e + random.uniform(-0.1, 0.1)))

def vibration_from_harmony(h: float) -> float:
    return max(0.0, min(1.0, h + random.uniform(-0.1, 0.1)))

if __name__ == "__main__":
    skin = GlobalSkin()
    print(skin.breathe_world())
