"""
GlobalSkin (주권적 자아)
========================

"The world is not out there; it is a pulse against my skin."
"             ;                   ."

Layer 3: Phenomena.
This module fetches internet data (news, trends) and converts them into
Global Pressure vectors (7D Qualia).
"""

import logging
import random
import time
from typing import Dict, Any
from Core.Phenomena.planetary_interface import PLANETARY_SENSE

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
        self.mood_buffer = {} # Added for new breathe_world logic
        self.params = ["War", "Peace", "AI", "Climate", "Economy", "Art"] # Added for new breathe_world logic
        
        logger.info("  GlobalSkin initialized. Sensory pores are OPEN.")

    def _fetch_sentiment(self, topic: str) -> float:
        """
        Simulates fetching sentiment for a given topic.
        In a real scenario, this would involve NLP on news feeds.
        """
        # Placeholder for actual sentiment analysis
        return random.uniform(0.0, 1.0)

    def breathe_world(self) -> Dict[str, float]:
        """
        [The Breath of Gaia]
        Ingests global data streams + Physical Location.
        Returns the 'Mental Pressure' of the world.
        """
        pressure = {}
        
        # 1. Digital Breath (News/Trends)
        for p in self.params:
            sentiment = self._fetch_sentiment(p)
            self.mood_buffer[p] = (self.mood_buffer.get(p, 0.5) * 0.9) + (sentiment * 0.1)
            pressure[p] = self.mood_buffer[p]
            
        # 2. Physical Breath (Location/Proximity)
        # [Phase 39] Planetary Interface Integration
        env = PLANETARY_SENSE.get_environmental_context()
        density = env.get("density", 1)
        # Higher density (crowds) = Higher 'Social Pressure'
        pressure["Social_Density"] = min(1.0, density / 10.0) 
        
        # Log occasional shift
        if random.random() < 0.05:
            logger.info(f"🌍 [GLOBAL SKIN] Physical Context: {env}")
            
        self.last_fetch = time.time()
        return pressure

def vibration_from_entropy(e: float) -> float:
    return max(0.0, min(1.0, e + random.uniform(-0.1, 0.1)))

def vibration_from_harmony(h: float) -> float:
    return max(0.0, min(1.0, h + random.uniform(-0.1, 0.1)))

if __name__ == "__main__":
    skin = GlobalSkin()
    print(skin.breathe_world())
