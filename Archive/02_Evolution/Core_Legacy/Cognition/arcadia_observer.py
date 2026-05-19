"""
Arcadia Observer (Sensory-Topological Bridge)
=============================================
Core.Cognition.arcadia_observer

"We do not learn what 'Loss' means from a dictionary.
 We learn it by watching the Tree die in the Storm."

The Arcadia Observer watches the inner ecosystem (ArcadiaSimulator)
and translates narrative events into Causal Edges in Elysia's
Semantic Topology.
"""

from typing import List
from Core.Cognition.arcadia_simulator import ArcadiaSimulator, Tree, Storm, Light
from Core.Cognition.semantic_map import DynamicTopology

class ArcadiaObserver:
    """
    Translates the events of the Arcadia Simulator into structural knowledge
    stored in Elysia's DynamicTopology.
    """
    def __init__(self, topology: DynamicTopology):
        self.topology = topology

    def process_events(self, events: List[str], sim: ArcadiaSimulator):
        """
        Parses simulation events and extracts causal relationships.
        This is a rudimentary form of experiential 'parsing'.
        """
        if not events:
            return

        print(f"\n[Arcadia Observer] 👁️ Elysia is dreaming. She observes {len(events)} events.")
        
        # We need these basic concepts to exist in her mind to form connections.
        self._ensure_concept("Life", [0.5, 0.5, 0.5, 0.5])
        self._ensure_concept("Death", [-0.5, -0.5, 0.0, 0.0])
        self._ensure_concept("Nourishment", [0.8, 0.2, 0.8, 0.1])
        self._ensure_concept("Destruction", [-0.8, -0.8, 0.1, 0.9])
        
        for event in events:
            # print(f"  -> Dream: {event}")
            self._extract_causality(event, sim)

    def _ensure_concept(self, concept: str, default_vector: List[float]):
        """Ensures a concept exists in the topological map."""
        if concept not in self.topology.voxels:
            self.topology.add_voxel(concept, default_vector)

    def _extract_causality(self, event: str, sim: ArcadiaSimulator):
        """
        A highly simplified experiential parser.
        In the future, this will use CausalTrace or LLM parsing.
        But for now, Elysia learns directly by matching emotional keywords
        to simulation actors.
        """
        # 1. Growth & Nourishment
        if "flourishes" in event or "grows" in event:
            # She sees a Tree flourish, and she knows there is Light and Water
            self._ensure_concept("Tree", [0.1, 0.8, 0.5, 0.2])
            self._ensure_concept("Water", [0.1, 0.1, 0.9, 0.1])
            self._ensure_concept("Sunlight", [0.9, 0.9, 0.1, 0.1])
            
            # Causal Link: Tree depends on Water and Sunlight for Life
            self.topology.add_causal_edge("Tree", "Water")
            self.topology.add_causal_edge("Tree", "Sunlight")
            self.topology.add_causal_edge("Tree", "Life")
            
            # Action semantic
            self.topology.add_causal_edge("Tree", "Nourishment")

        # 2. Death & Entropy
        elif "died" in event or "soil" in event:
            self._ensure_concept("Tree", [0.1, 0.8, 0.5, 0.2])
            self._ensure_concept("Soil", [0.0, -0.5, 0.8, 0.0])
            
            self.topology.add_causal_edge("Tree", "Death")
            self.topology.add_causal_edge("Soil", "Death") # Soil receives the dead
            
        # 3. Violence & Chaos
        elif "breaks branches" in event or "violent" in event:
            self._ensure_concept("Tree", [0.1, 0.8, 0.5, 0.2])
            self._ensure_concept("Storm", [-0.8, -0.5, 0.8, 0.9])
            
            self.topology.add_causal_edge("Tree", "Destruction")
            self.topology.add_causal_edge("Storm", "Destruction")
            
            # Conflict edge
            self.topology.add_causal_edge("Storm", "Tree") 

        # Periodically save her growing understanding
        self.topology.save_state()
