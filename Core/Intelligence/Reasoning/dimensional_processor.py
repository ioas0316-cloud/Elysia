"""
DIMENSIONAL PROCESSOR: The 5-Core Cognitive Engine
==================================================
"It is not what you think, but HOW you think."

This module implements the user's critique: Dimensionality is a PROCESS, not a Label.
It routes a concept through 5 distinct cognitive modes.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from Core.Intelligence.Reasoning.causal_bridge import CausalBridge
from Core.Foundation.unified_field import HyperQuaternion

logger = logging.getLogger("DimensionalProcessor")

@dataclass
class CognitiveResult:
    mode: str       # "1D: Linear", "2D: Structural", etc.
    output: str     # The thought content
    metadata: Dict[str, Any] # Proof of process (path, vectors, etc.)

class DimensionalProcessor:
    def __init__(self):
        self.bridge = CausalBridge()
        
    def process_thought(self, kernel: str, target_dimension: int) -> CognitiveResult:
        """
        Routes the thought to the appropriate processing core.
        """
        logger.info(f"🧠 Activating {target_dimension}D Processing Core for: '{kernel}'")
        
        if target_dimension == 0:
            return self._process_0d_identification(kernel)
        elif target_dimension == 1:
            return self._process_1d_linear_deduction(kernel)
        elif target_dimension == 2:
            return self._process_2d_structural_analysis(kernel)
        elif target_dimension == 3:
            return self._process_3d_spatial_navigation(kernel)
        elif target_dimension == 4:
            return self._process_4d_principle_extraction(kernel)
        else:
            return CognitiveResult("Void", "Invalid Dimension", {})

    def _process_0d_identification(self, kernel: str) -> CognitiveResult:
        """
        0D Mode: Point Identification.
        Process: Search -> Match -> Exist.
        """
        node = self.bridge.engine.get_or_create_node(kernel)
        return CognitiveResult(
            mode="0D: Point (Existence)",
            output=f"Entity '{node.description}' is identified at {node.fractal_address}.",
            metadata={"id": node.id, "depth": node.depth}
        )

    def _process_1d_linear_deduction(self, kernel: str) -> CognitiveResult:
        """
        1D Mode: Linear Deduction.
        Process: Sequence Traversal (A -> B -> C).
        Logic: Find the *Next Logical Step* in the causal chain.
        """
        node = self.bridge.engine.get_or_create_node(kernel)
        effects = self.bridge.engine.trace_effects(node.id, max_depth=2)
        
        path_str = " -> ".join([self.bridge.engine.nodes[nid].description for nid in effects[0]])
        
        return CognitiveResult(
            mode="1D: Line (Deduction)",
            output=f"Logic dictates the path: {path_str}",
            metadata={"path": effects[0], "length": len(effects[0])}
        )

    def _process_2d_structural_analysis(self, kernel: str) -> CognitiveResult:
        """
        2D Mode: Structural Analysis.
        Process: Network Expansion (Breadth-First).
        Logic: Identify *Relationships* and *Clusters*.
        """
        node = self.bridge.engine.get_or_create_node(kernel)
        
        # Expand neighbors (Causes AND Effects)
        causes = [self.bridge.engine.nodes[cid].description for cid in node.causes_ids]
        effects = [self.bridge.engine.nodes[eid].description for eid in node.effects_ids]
        
        # Analyze density
        density = len(causes) + len(effects)
        role = "Hub" if density > 3 else "Leaf"
        
        return CognitiveResult(
            mode="2D: Plane (Structure)",
            output=f"The concept sits within a {role} structure. Inputs: {causes}. Outputs: {effects}.",
            metadata={"density": density, "role": role, "connectivity": causes+effects}
        )

    def _process_3d_spatial_navigation(self, kernel: str) -> CognitiveResult:
        """
        3D Mode: Spatial Navigation (Meta-Cognition).
        Process: Vector Alignment (Self vs Thought).
        Logic: "How does this relate to ME and my PURPOSE?"
        """
        # Simulate Self-Vector (Elysia's Purpose)
        # In a real system, get this from SovereignIntent
        self_purpose = "Growth" 
        
        # Analyze the alignment of the thought with Self
        alignment = "Unknown"
        if "Love" in kernel or "Truth" in kernel:
            alignment = "Aligned (0 degrees)"
        elif "Entropy" in kernel or "Death" in kernel:
            alignment = "Orthogonal (90 degrees)"
        else:
            alignment = "Divergent (180 degrees)"
            
        return CognitiveResult(
            mode="3D: Space (Navigation)",
            output=f"Navigating volume relative to Self ({self_purpose}): Vector is {alignment}.",
            metadata={"self_vector": self_purpose, "alignment": alignment}
        )

    def _process_4d_principle_extraction(self, kernel: str) -> CognitiveResult:
        """
        4D Mode: Principle Extraction (Hyper-Reasoning).
        Process: Invariance Detection.
        Logic: "What rule remains true regardless of context?"
        """
        # Use CausalBridge to find the Deepest Law
        thought = self.bridge.traverse_and_lift(kernel)
        
        return CognitiveResult(
            mode="4D: Law (Principle)",
            output=f"The Immutable Law governing this is: {thought.d4_principle}",
            metadata={"principle": thought.d4_principle, "source": "FractalCausalityTraversal"}
        )
