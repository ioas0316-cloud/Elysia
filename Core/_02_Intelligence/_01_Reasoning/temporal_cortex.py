
import logging
from typing import List, Dict, Any, Tuple
import time

logger = logging.getLogger("TemporalCortex")

class TemporalCortex:
    """
    The Narrative Weaver.
    
    Transforms the 'Soup of Now' (Active Context) into a 'Thread of Time' (Narrative).
    """
    
    def __init__(self, internal_universe):
        self.universe = internal_universe
        logger.info("⏳ TemporalCortex initialized - The Storyteller awakens.")
        
    def weave_narrative(self) -> str:
        """
        Weaves active concepts into a narrative context string.
        Used to give the LLM/Brain existing state awareness.
        """
        # 1. Get Active Concepts (Soup)
        context_dict = self.universe.get_active_context(limit=10)
        
        if not context_dict:
            return "Mind is silent."
            
        # 2. Enrich with Timestamp data (if available in coordinate map)
        narrative_nodes = []
        for name in context_dict.keys():
            coord = self.universe.coordinate_map.get(name)
            if coord:
                narrative_nodes.append({
                    "name": name,
                    "depth": coord.depth,
                    "time": getattr(coord, 'timestamp', 0),
                    "frequency": coord.frequency
                })

        # 3. Sort by Time (Chronological Narrative)
        # Oldest first -> Thread of events
        narrative_nodes.sort(key=lambda x: x['time'])
        
        # 4. Construct Narrative String
        narrative = "Recent Context (Temporal Stream):\n"
        now = time.time()
        
        for node in narrative_nodes:
            age = now - node['time']
            age_str = f"{age:.0f}s ago"
            narrative += f"- [{age_str}] {node['name']} (Depth: {node['depth']:.2f})\n"
            
        return narrative

    def get_context_vector(self):
        """
        Returns a weighted vector representation of the current moment.
        (For future integration with TorchGraph)
        """
        pass
