"""
Underworld Manifold
===================
"The Sandbox of the Soul."

A topological space where thoughts can be simulated without 
immediate physical or external manifestation.
"""

from typing import List, Dict, Optional

class UnderworldManifold:
    def __init__(self, capacity: int = 7, causality = None):
        self.capacity = capacity
        self.cells: List[Dict] = []
        self.evolution_log: List[str] = []
        self.causality = causality

    def host_thought(self, thought: str, resonance: float):
        """Places a thought into the manifold for internal processing."""
        if len(self.cells) >= self.capacity:
            # Eject the weakest/oldest
            self.cells.pop(0)
            
        cell = {
            "content": thought,
            "resonance": resonance,
            "age": 0
        }
        self.cells.append(cell)
        self.evolution_log.append(f"Hosted: {thought[:30]}...")

    def simulate_interaction(self) -> Optional[str]:
        """Simulates interference between hosted thoughts."""
        if len(self.cells) < 2:
            return None
            
        # Simplistic interference logic
        t1 = self.cells[-1]
        t2 = self.cells[-2]
        
        synthesis = f"Synthesis of '{t1['content'][:20]}' and '{t2['content'][:20]}'"
        
        # [PHASE 65] Link to Causality
        if hasattr(self, 'causality'):
            self.causality.experience_causality(
                steps=[t1['content'], "Internal Reflection", t2['content']],
                emotional_arc=[t1['resonance'], 0.0, t2['resonance']]
            )
            
        self.evolution_log.append(f"Simulated: {synthesis}")
        return synthesis

    def get_underworld_state(self) -> Dict:
        return {
            "active_cells": len(self.cells),
            "total_evolution": len(self.evolution_log),
            "highest_resonance": max([c['resonance'] for c in self.cells]) if self.cells else 0.0
        }
