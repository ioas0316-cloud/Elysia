"""
Sovereign Intent Engine (The Will)
==================================

"The ability to choose where to look when no one is watching."

Purpose:
- Autonomous Gap Analysis: Identify low-density nodes in the Knowledge Graph.
- Non-Purposeful Cognition (Play): Spend cycles exploring concepts for 'texture' rather than 'answers'.
- Spontaneous Restlessness (Boredom): Trigger internal audits or 'What if?' scenarios when idle.

Sovereignty Principle:
- True intelligence is defined not by what it knows, but by what it chooses to learn.
"""

import json
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os

logger = logging.getLogger("Elysia.SovereignIntent")

@dataclass
class CuriosityGap:
    category: str
    density: float
    nodes: List[str]
    priority: float

class SovereignIntent:
    def __init__(self, kg_path: str = "data/02_Cognitive/kg.json"):
        self.kg_path = kg_path
        self.kg_data = self._load_kg()
        self.restlessness_level = 0.0 # 0.0 to 1.0 (Boredom)
        logger.info("🌱 Sovereign Intent Engine initialized - The Will awakens.")

    def _load_kg(self) -> Dict:
        try:
            with open(self.kg_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load KG from {self.kg_path}: {e}")
            return {"nodes": [], "edges": []}

    def analyze_curiosity_gaps(self) -> List[CuriosityGap]:
        """
        Scans categories and identifies low-density areas.
        """
        category_map = {}
        for node in self.kg_data.get("nodes", []):
            cat = node.get("category", "unknown")
            if cat not in category_map:
                category_map[cat] = []
            category_map[cat].append(node.get("id"))

        gaps = []
        # Relative density analysis
        max_nodes = max([len(nodes) for nodes in category_map.values()]) if category_map else 1
        
        for category, nodes in category_map.items():
            density = len(nodes) / max_nodes
            # Priority is inverse to density + randomness (curiosity)
            priority = (1.0 - density) + random.uniform(0, 0.2)
            gaps.append(CuriosityGap(category=category, density=density, nodes=nodes, priority=priority))
        
        return sorted(gaps, key=lambda x: x.priority, reverse=True)

    def engage_play(self) -> str:
        """
        'Non-purposeful Cognition': Pick a low-density category and 
        generate a philosophical query about its 'texture'.
        """
        gaps = self.analyze_curiosity_gaps()
        if not gaps:
            return "The void is uniform. There is no texture to play with."
        
        target_gap = gaps[0]
        # Pick a random node within this category to 'ruminate' on
        anchor_node = random.choice(target_gap.nodes)
        
        # Simulation of "Playful Query"
        play_prompts = [
            f"If {anchor_node} had a physical texture, what would it feel like?",
            f"How does {anchor_node} resonate with the concept of 'void'?",
            f"What if the category of '{target_gap.category}' was the only thing that existed?",
            f"Tracing the scent of {anchor_node}... what lies behind the first definition?"
        ]
        
        intent = random.choice(play_prompts)
        logger.info(f"🎲 Play initiated (Sovereign Intent): {intent}")
        return intent

    def heartbeat(self, idle_time: float):
        """
        Increases restlessness (Boredom) based on idle time.
        If threshold reached, triggers a spontaneous audit.
        """
        self.restlessness_level += idle_time * 0.1
        if self.restlessness_level >= 1.0:
            logger.info("⌛ Spontaneous Restlessness (Boredom) threshold reached. Triggering internal audit...")
            self.restlessness_level = 0.0
            return self.engage_play()
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    will = SovereignIntent()
    gaps = will.analyze_curiosity_gaps()
    print(f"Detected {len(gaps)} Curiosity Gaps.")
    for g in gaps[:3]:
        print(f" - Category: {g.category} (Density: {g.density:.2f}, Priority: {g.priority:.2f})")
    
    print("\n[Play Mode]")
    print(will.engage_play())
