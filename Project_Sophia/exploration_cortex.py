import random
from tools.kg_manager import KGManager
from typing import Dict, Optional

class ExplorationCortex:
    """
    Autonomously explores the knowledge graph to discover new or interesting concepts,
    triggering emotional responses and learning opportunities.
    """
    def __init__(self, kg_manager: KGManager):
        self.kg_manager = kg_manager

    def discover_random_concept(self) -> Optional[Dict]:
        """
        Selects a random node from the knowledge graph as a "discovery."
        """
        nodes = self.kg_manager.kg.get('nodes')
        if not nodes:
            return None

        discovered_concept = random.choice(nodes)
        print(f"[ExplorationCortex] Today's discovery: {discovered_concept.get('id')}")
        return discovered_concept
