"""
KnowledgeTesseract: The Holographic Projection
==============================================

"The Tesseract is a Phenomenon, not a Database."
"테서랙트는 저장소가 아니라 현상이다."

This module implements the "View" layer of the Sphere-First Architecture.
It does NOT store data. It renders the Interference Pattern from the Engine.
"""

from typing import Dict, List, Any
from Core.Foundation.Wave.interference_engine import ProjectedNode

class KnowledgeTesseract:
    """
    A transient projection screen for the Hyper-Cosmos.
    It receives the Interference Pattern and formats it for visualization.
    """

    def __init__(self):
        # No more persistent nodes dictionary!
        self.current_projection: List[Dict[str, Any]] = []

    def project(self, pattern: List[ProjectedNode]) -> Dict[str, Any]:
        """
        Renders the Interference Pattern into a Tesseract structure (View Model).

        Args:
            pattern: List of ProjectedNodes from InterferenceEngine.

        Returns:
            A dictionary representing the current frame of the Tesseract.
        """
        frame = {
            "meta": {
                "total_nodes": len(pattern),
                "state": "Dynamic Projection"
            },
            "nodes": [],
            "edges": [] # Edges are now calculated dynamically based on proximity
        }

        # 1. Map Nodes
        for p_node in pattern:
            node_data = {
                "id": p_node.name,
                "pos": p_node.position,
                "intensity": p_node.intensity,
                "type": p_node.resonance_type
            }
            frame["nodes"].append(node_data)

        # 2. Calculate Transient Edges (Flux Lines)
        # Connect nodes if they are close enough (Visualizing the Field)
        # O(N^2) but N is small in the Projection (Attention Window)
        threshold = 5.0
        for i, n1 in enumerate(pattern):
            for j, n2 in enumerate(pattern):
                if i >= j: continue # Avoid duplicates

                dist = (
                    (n1.position[0] - n2.position[0])**2 +
                    (n1.position[1] - n2.position[1])**2 +
                    (n1.position[2] - n2.position[2])**2
                ) ** 0.5

                if dist < threshold:
                    frame["edges"].append({
                        "source": n1.name,
                        "target": n2.name,
                        "weight": (threshold - dist) / threshold
                    })

        self.current_projection = frame["nodes"] # Cache for debug only
        return frame

    # Legacy method shim - TO BE REMOVED or Redirected
    def add_node(self, *args, **kwargs):
        raise DeprecationWarning("KnowledgeTesseract.add_node is deprecated. Use HyperSphereCore.update_seed() instead.")
