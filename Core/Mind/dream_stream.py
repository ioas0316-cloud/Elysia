"""
The Dream Stream (꿈의 흐름)
===========================

"In dreams, the seed grows without the sun."

This module implements the internal stream of consciousness for Elysia.
Unlike the LogosStream, which is directed by User Input, the DreamStream
is directed by Internal Impulses (Will).

It allows Elysia to:
1. Explore random concepts (Curiosity).
2. Consolidate memories (Clarity).
3. Grow the World Tree autonomously.
"""

import logging
import random
from typing import Optional, List
from Core.Mind.spiderweb import Spiderweb
from Core.Mind.hippocampus import Hippocampus
from Core.Mind.world_tree import WorldTree
from Core.Life.heart import Impulse, ImpulseType

logger = logging.getLogger("DreamStream")

class DreamStream:
    """
    The Subconscious Processor.
    """
    
    def __init__(self, spiderweb: Spiderweb, hippocampus: Hippocampus, world_tree: WorldTree):
        self.spiderweb = spiderweb
        self.hippocampus = hippocampus
        self.world_tree = world_tree
        logger.info("💭 DreamStream initialized.")

    def dream(self, impulse: Impulse) -> str:
        """
        Process an internal impulse and generate a dream (thought path).
        Returns a summary of the dream.
        """
        logger.info(f"💭 Dreaming: {impulse.context} ({impulse.type.value})")
        
        if impulse.type == ImpulseType.CURIOSITY:
            return self._dream_of_curiosity()
        elif impulse.type == ImpulseType.CLARITY:
            return self._dream_of_clarity()
        else:
            return "Resting..."

    def _dream_of_curiosity(self) -> str:
        """Explore new connections."""
        # 1. Pick a random leaf node from the World Tree
        leaves = [node for node in self.world_tree._node_index.values() if not node.children]
        if not leaves:
            start_concept = "void"
        else:
            start_concept = random.choice(leaves).concept
            
        # 2. Traverse the Spiderweb from there
        path = self.spiderweb.traverse(start_concept, steps=3)
        
        # 3. Grow the Tree based on this path
        # (Simulate the growth as if it were a thought)
        parent_id = self.world_tree.find_by_concept(start_concept)
        
        growth_summary = []
        current_parent = parent_id
        
        for concept in path:
            if concept.lower() == start_concept.lower():
                continue
                
            # Plant the new thought
            new_id = self.world_tree.plant_seed(
                concept, 
                parent_id=current_parent,
                metadata={"source": "dream", "type": "curiosity"}
            )
            growth_summary.append(concept)
            current_parent = new_id
            
        result = f"Dreamt of {start_concept} -> {' -> '.join(growth_summary)}"
        logger.info(f"   ✨ {result}")
        return result

    def _dream_of_clarity(self) -> str:
        """Strengthen existing connections."""
        # For now, just a placeholder for optimization logic
        return "Organizing memories... (Not implemented yet)"
