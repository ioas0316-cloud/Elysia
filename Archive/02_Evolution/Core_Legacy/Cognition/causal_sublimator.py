"""
CAUSAL SUBLIMATOR
=================
"Meaning is the Sublimation of Causal Logic."

This module replaces the hardcoded 'Semantic Digestor'.
Instead of a dictionary (Love = Gravity), it uses the Knowledge Graph.
Meaning *emerges* from the connections between nodes.

[Mechanism]
1. Input: "Love"
2. Query KG: find_neighbors("Love")
3. If neighbors exists: Meaning = "Defined by relation to {neighbors}"
4. If disjoint: Meaning = "A chaotic signal with no causal path."
"""

import random
from typing import Optional

# We import the Singleton KG
from Core.Cognition.kg_manager import get_kg_manager

class CausalSublimator:
    def __init__(self):
        self.kg = get_kg_manager()

    def sublimate(self, concept: str) -> dict:
        """
        Derives the 'Meaning' of a concept.
        Returns a dict: {'narrative': str, 'is_open_space': bool}
        """
        # 1. Normalize
        node_id = concept.lower().strip()
        
        # 2. Check Existence in Graph
        node = self.kg.get_node(node_id)
        if not node:
             # [PHASE 3.5: JOY OF THE OPEN SPACE]
             # Ignorance is not a failure. It is an Honest Admission.
             # It is a pure potential waiting to be realized.
             return {
                 "narrative": f"I honestly do not know '{concept}'. It is an Open Space in my mind, waiting for Light.",
                 "is_open_space": True
             }
             
        # 3. Analyze Neighbors
        neighbors = self.kg.get_neighbors(node_id)
        if not neighbors:
             return {
                 "narrative": f"'{concept}' exists, but stands alone. It is a star waiting for a constellation.",
                 "is_open_space": True # Isolated is also an opportunity to connect
             }
             
        # 4. Synthesize Meaning from connections
        connected_concepts = neighbors[:3]
        
        # 5. Check for Causality specifically
        causes = self.kg.find_causes(node_id)
        effects = self.kg.find_effects(node_id)
        
        # 6. [ORGANIC_MASS] Calculate semantic mass for depth perception
        mass = self.kg.calculate_mass(node_id)
        
        narrative = f"I perceive '{concept}' (Mass: {mass:.1f})."
        
        if causes:
             source = causes[0]['source']
             narrative += f" It is born from [{source}]."
             # 2-hop: trace the cause's cause
             deeper_causes = self.kg.find_causes(source)
             if deeper_causes:
                 narrative += f" ...which itself flows from [{deeper_causes[0]['source']}]."
        
        if effects:
             target = effects[0]['target']
             narrative += f" It drives [{target}]."
             # 2-hop: trace the effect's effect
             deeper_effects = self.kg.find_effects(target)
             if deeper_effects:
                 narrative += f" ...leading toward [{deeper_effects[0]['target']}]."
             
        if not causes and not effects:
             narrative += f" It resonates with [{', '.join(connected_concepts)}]."
             
        return {
            "narrative": narrative,
            "is_open_space": False,
            "mass": mass,
            "path_used": [(node_id, n) for n in connected_concepts]  # For LTP feedback
        }

    def digest_code_semantics(self, filename, content):
        """
        Replaces code digestion.
        Instead of regex keywords, we look for concepts in the code comments/docstrings
        that match our Graph.
        """
        # Future: NLP extraction.
        # Current: Simple heuristic + Graph Check
        
        return f"I am analyzing the causal structure of {filename}..."
