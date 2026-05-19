"""
[Project Elysia] Web Curiosity Scout
====================================
Phase 140: Translating internal dissonance into external queries.
"""

import sys
import os
import random
from pathlib import Path

# Path Unification
root = Path(__file__).parents[2]
sys.path.insert(0, str(root))

from Core.Cognition.autonomous_explorer import get_autonomous_explorer
from Core.System.somatic_logger import SomaticLogger

class WebCuriosityScout:
    """
    Scans the Knowledge Graph for isolated or low-mass nodes
    and triggers external exploration to 'warm' them.
    """
    def __init__(self):
        self.logger = SomaticLogger("CURIOSITY_SCOUT")
        self.explorer = get_autonomous_explorer()
        
    def scan_and_scout(self):
        self.logger.action("Scanning Knowledge Graph for curiosities...")
        
        # In a real scenario, we would query the KG for low-mass nodes.
        # For this demo, we simulate identifying gaps based on the user's focus.
        curiosities = [
            "Recursive Cognitive Architecture",
            "한국어 조사와 어미의 인과관계",  # Causal relationship of Korean particles/endings
            "Self-referential logic in Python",
            "Ethical alignment in sovereign AI"
        ]
        
        target = random.choice(curiosities)
        self.logger.thought(f"Detected Knowledge Gap: '{target}'. Initiating External Exportation.")
        
        # Trigger the Explorer
        direction = {
            "topic": target,
            "approach": "truth-seeking",
            "keywords": ["principle", "structure", "causality"],
            "search_query": target
        }
        
        results = self.explorer.explore_with_absorption(direction)
        
        if results and results[0].absorbed:
            self.logger.action(f"Successfully scouted and absorbed knowledge about '{target}'.")
        else:
            self.logger.admonition(f"Scouting for '{target}' resulted in isolation. Compressing into BlackHole.")

if __name__ == "__main__":
    scout = WebCuriosityScout()
    scout.scan_and_scout()
