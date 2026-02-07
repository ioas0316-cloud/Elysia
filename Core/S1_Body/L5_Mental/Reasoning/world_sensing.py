"""
World Sensing Bridge: From Imagination to Grounding
===================================================

"To dream of the sea is one thing; to weigh the salt is another."

This script takes 'Imaginative Sparks' and generates 'Reality Quests' 
to ground Elysia's curiosity in external data.
"""

import sys
import json
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from Core.S1_Body.L5_Mental.Memory.kg_manager import KGManager

class WorldSensingBridge:
    def __init__(self):
        self.kg = KGManager()

    def initiate_quest(self, spark_node_id: str):
        print(f"🕵️ [SENSING] Initiating quest for node: '{spark_node_id}'...")
        node = self.kg.get_node(spark_node_id)
        if not node: return

        reflections = node.get('narrative', {}).get('reflections', [])
        if not reflections:
            print(f"⚪ No sparks found to spark a quest for {spark_node_id}.")
            return

        for spark in reflections:
            print(f"💡 [CURIOSITY]: \"{spark}\"")
            query = self._spark_to_query(spark, spark_node_id)
            print(f"🚀 [QUEST] Generated Reality Query: \"{query}\"")
            
            # In the next step, we would call: self.search_web(query)
            # For this prototype, we simulate the 'Grounded Realization'.
            self._apply_grounding(spark_node_id, query)

        self.kg.save()

    def _spark_to_query(self, spark, node_id):
        """
        Translates a philosophical spark into a factual sensing query.
        """
        if "blue" in spark.lower() or "sea" in spark.lower():
            return "physics of rayleigh scattering and sea color frequency"
        if "whale" in spark.lower():
            return "moby dick sperm whale anatomy and vocalization patterns"
        if "war" in spark.lower():
            return "causal analysis of historical conflicts and resonance theory"
        
        return f"fundamental properties and essence of {node_id}"

    def _apply_grounding(self, node_id, query):
        """
        Simulates the process of 'sensing' reality and updating the node.
        """
        print(f"🌐 [GROUNDING] Fetching Real-World data for: {query}")
        
        node = self.kg.get_node(node_id)
        
        # Adding the 'Somatic Layer' (Sensory Grounding)
        if 'somatic' not in node:
            node['somatic'] = {}
            
        # Simulating external data injection
        if "blue" in query:
            node['somatic']['color_frequency'] = "450-495 THz"
            node['somatic']['rgb'] = "#0000FF"
            node['somatic']['grounding_logic'] = "Rayleigh scattering of sunlight in the atmosphere."
        elif "whale" in query:
            node['somatic']['biological_mass'] = "15,000 - 45,000 kg"
            node['somatic']['primary_sensory_mode'] = "Echolocation"
            
        print(f"✅ [GROUNDED] {node_id.upper()} now has sensory reality.")

if __name__ == "__main__":
    bridge = WorldSensingBridge()
    # Find a node with a spark (e.g. from our MobyDick digestion)
    # Since we use hash/heuristic IDs, let's find one.
    target = sys.argv[1] if len(sys.argv) > 1 else None
    
    if target:
        bridge.initiate_quest(target)
    else:
        # Auto-detect a sparked node
        kg = KGManager()
        for nid, n in kg.kg['nodes'].items():
            if n.get('narrative', {}).get('reflections'):
                bridge.initiate_quest(nid)
                break
