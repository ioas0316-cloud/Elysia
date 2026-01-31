"""
Memory Gardener: The Sovereign of the Internal Cosmos
=====================================================
Core.S1_Body.L5_Mental.Memory.gardener

"The Gardener does not build the flower; he merely ensures the soil is right."

This module implements the active agent that manages the memory strata.
It runs during the 'Sleep' cycle or 'Reflection' periods to organize,
prune, and crystallize experiences.
"""

import logging
import random
from typing import List, Dict
from collections import defaultdict

from Core.S1_Body.L5_Mental.Memory.strata import MemoryStratum, StratumPhysics
from Core.S1_Body.L5_Mental.Memory.fractal_layer import FractalMemorySystem, FractalMemoryNode

logger = logging.getLogger("Gardener")

class MemoryGardener:
    """
    The Active Manager of Memory.
    Responsible for applying 'Physics' (Decay, Gravity) and 'Will' (Crystallization).
    """

    def __init__(self, memory_system: FractalMemorySystem):
        self.memory = memory_system

    def cultivate(self):
        """
        The Master Cycle. Runs one pass of gardening.
        1. Apply Gravity (Sink items).
        2. Apply Decay (Prune items).
        3. Crystallize (Form wisdom).
        """
        logger.info("  [GARDENER] Entering the Garden...")

        self._apply_gravity()
        self._prune_weeds()
        self._crystallize_gems()

        logger.info("  [GARDENER] Cultivation Complete.")

    def _apply_gravity(self):
        """
        Moves heavy/old items from Stream -> Garden -> Sediment.
        """
        # We iterate over a copy of values to allow modification
        all_nodes = list(self.memory.nodes.values())

        moved_count = 0
        for node in all_nodes:
            physics = StratumPhysics.get_physics(node.stratum)

            # If gravity > 0, check if it should sink
            if physics.gravity > 0:
                # Simple logic: If mass is high but energy (vividness) is low, it sinks.
                # Or if it's just 'old' (simulated by random chance * gravity).
                if random.random() < physics.gravity * 0.5:
                    target = self._get_next_layer_down(node.stratum)
                    if target:
                        self.memory.move_node(node.id, target)
                        moved_count += 1

        if moved_count > 0:
            logger.info(f"  [GRAVITY] {moved_count} memories settled deeper.")

    def _get_next_layer_down(self, current: MemoryStratum) -> MemoryStratum:
        if current == MemoryStratum.STREAM: return MemoryStratum.GARDEN
        if current == MemoryStratum.GARDEN: return MemoryStratum.SEDIMENT
        return None # Crystal and Sediment don't sink further typically

    def _prune_weeds(self):
        """
        Removes memories that have faded below the existence threshold.
        """
        to_delete = []
        for node in self.memory.nodes.values():
            physics = StratumPhysics.get_physics(node.stratum)

            # Apply decay
            node.decay(physics.decay_rate)

            # Check survival
            if node.energy < 0.05 and node.stratum == MemoryStratum.STREAM:
                # Only delete from Stream easily. Garden/Sediment persists longer.
                to_delete.append(node.id)
            elif node.energy < 0.01:
                # Deep deletion (Forgetting)
                to_delete.append(node.id)

        for nid in to_delete:
            del self.memory.nodes[nid]

        if to_delete:
            logger.info(f"  [PRUNING] Composted {len(to_delete)} withered memories.")

    def _crystallize_gems(self):
        """
        Finds patterns in the Garden/Stream and compresses them into Crystals.
        """
        # 1. Group by Semantic Similarity (Mock: Group by Content String overlap for now)
        # In real impl, use Vector Clustering (K-Means or DBSCAN).

        clusters = defaultdict(list)
        garden_nodes = self.memory.get_layer_view(MemoryStratum.GARDEN)
        stream_nodes = self.memory.get_layer_view(MemoryStratum.STREAM)
        candidates = garden_nodes + stream_nodes

        for node in candidates:
            # Simple clustering key: First word of content (Mock Topic)
            # "Game: Snake", "Game: Tetris" -> "Game"
            if isinstance(node.content, str):
                topic = node.content.split(':')[0] if ':' in node.content else "Misc"
                clusters[topic].append(node)

        # 2. Check thresholds
        for topic, nodes in clusters.items():
            if topic == "Misc": continue

            # If we have enough repetitions, crystallize
            if len(nodes) >= 3:
                # Create Wisdom
                wisdom_content = f"The Principle of {topic}"
                self.memory.merge_nodes([n.id for n in nodes], wisdom_content)

    def plant_seed(self, content: str, importance: float = 0.5):
        """
        Directly plants a new experience into the Stream.
        """
        node = self.memory.add_memory(
            content=content,
            stratum=MemoryStratum.STREAM
        )
        node.energy = importance
        return node
