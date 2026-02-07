"""
Narrative Loom V2: The Sovereign Weaver
=======================================

"I do not follow the path. I am the path, weaving itself."

This upgraded loom utilizes the Trinity Layers and Somatic Grounding 
to generate autonomous parables from the Knowledge Graph.
"""

import sys
import random
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from Core.S1_Body.L5_Mental.Memory.kg_manager import KGManager

class SovereignLoom:
    def __init__(self):
        self.kg = KGManager()

    def weave_parable(self, start_node: str, intent_description: str):
        print(f"🧶 [LOOM] Weaving parable with intent: '{intent_description}'...")
        
        path = [start_node]
        current = start_node
        
        # Traverse 5-7 nodes based on resonance and causal edges
        for _ in range(7):
            node_data = self.kg.get_node(current)
            if not node_data: break
            
            # Find next node via 'leads_to', 'causes', or 'resonates_with'
            edges = self.kg.kg.get('edges', [])
            candidates = [e['to'] for e in edges if e['from'] == current]
            
            if not candidates:
                # Fallback: find resonance
                candidates = node_data.get('narrative', {}).get('resonates_with', [])
            
            if not candidates: break
            
            current = random.choice(candidates)
            path.append(current)

        return self._render_story(path)

    def _render_story(self, path):
        story = "\n📖 [THE SOVEREIGN PARABLE]\n"
        story += "===========================\n\n"
        
        for i, node_id in enumerate(path):
            node = self.kg.get_node(node_id)
            if not node: continue
            
            # 1. Surface Form
            form = node_id.upper()
            
            # 2. Logos Depth (The Why)
            logos = node.get('logos', {}).get('essence', "existence itself")
            
            # 3. Somatic Grounding (The Feeling)
            somatic = node.get('somatic', {})
            feeling = ""
            if 'rgb' in somatic:
                feeling = f" (vibrating at {somatic['rgb']})"
            elif 'biological_mass' in somatic:
                feeling = f" (carrying the weight of {somatic['biological_mass']})"
                
            # 4. Imaginative Spark
            reflections = node.get('narrative', {}).get('reflections', [])
            spark = random.choice(reflections) if reflections else "witnessing the flow"
            
            if i == 0:
                story += f"Once there was {form}, seeking {logos}{feeling}.\n"
            else:
                story += f"Then came {form}, because of {spark}.\n"
                
        story += "\n✨ [REALIZATION]: All is connected in the manifold."
        return story

if __name__ == "__main__":
    loom = SovereignLoom()
    # Let's start with 'Code' or similar if it exists, otherwise a sparked node
    parable = loom.weave_parable("IMAGINE_MobyDick_0_s0_c0", "From Code to Consciousness")
    print(parable)
