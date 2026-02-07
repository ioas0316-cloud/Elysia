"""
Narrative Loom: The Weaver of Causal Stories
============================================

"A point is a fact; a line is a relation; a story is a field in motion."

This script traverses the Knowledge Graph using a 'Field of Intent' 
to generate autonomous causal narratives.
"""

import sys
import random
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from Core.S1_Body.L5_Mental.Memory.kg_manager import KGManager

class NarrativeLoom:
    def __init__(self):
        self.kg = KGManager()

    def weave(self, seed_node: str, intent: str = "expansion", depth: int = 5):
        """
        Weaves a story starting from a seed node, biased by an intent.
        Intents: expansion, contraction, synthesis, resonance.
        """
        current = seed_node.lower()
        if not self.kg.get_node(current):
            return f"The Loom cannot find the seed of '{current}'."

        narrative_path = [current]
        story_segments = []

        print(f"🧵 [LOOM] Weaving story from '{seed_node}' with intent '{intent}'...")

        for i in range(depth):
            node = self.kg.get_node(current)
            edges = [e for e in self.kg.kg.get('edges', []) if e['source'] == current]
            
            if not edges:
                break

            # [FIELD OF INTENT] Biasing the next step
            # For simplicity: 'expansion' favors 'resonates_with', 'synthesis' favors 'constitutes'
            if intent == "expansion":
                candidates = [e for e in edges if e['relation'] in ["resonates_with", "linked_to"]]
            elif intent == "synthesis":
                # Look into the inner_cosmos for constitutive logic
                cosmos = node.get("inner_cosmos", {})
                if cosmos and cosmos.get("nodes"):
                    inner_nodes = list(cosmos['nodes'].keys())
                    current = random.choice(inner_nodes)
                    narrative_path.append(current)
                    story_segments.append(f"Diving deeper, we find {current} at the core.")
                    continue
                candidates = edges
            else:
                candidates = edges

            if not candidates:
                candidates = edges # Fallback

            next_edge = random.choice(candidates)
            current = next_edge['target']
            narrative_path.append(current)

        # Generating the "Morphological Narrative"
        story = self._synthesize_narrative(narrative_path, intent)
        return story

    def _synthesize_narrative(self, path, intent):
        """
        Transforms a path of nodes into a 'Living Narrative'.
        """
        prompt = f"Field of Intent: {intent}\nPath: {' -> '.join(path)}\n\n--- CAUSAL NARRATIVE ---\n"
        
        # Heuristic Narrative Generation (In Phase 9, this will use LogosLLM)
        segments = []
        for i in range(len(path) - 1):
            n1 = path[i]
            n2 = path[i+1]
            segments.append(f"From the essence of {n1.upper()}, the resonance shifts toward {n2.upper()}.")
        
        segments.append(f"\nThus, through the law of {intent}, the manifold realizes his new state.")
        return prompt + " ".join(segments)

if __name__ == "__main__":
    loom = NarrativeLoom()
    # Let's weave a story of 'Water' seeking 'Logic'
    print(loom.weave("water", "expansion"))
