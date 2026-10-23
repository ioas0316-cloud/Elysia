"""
Visual Causal Cognition Pipeline (VisualCausalCognitionPipeline)
==================================================================
End-to-End Autonomous Pipeline combining:
1. CausalGraphExtractor (2D Pixel Stream -> Nodes/Edges/Constraints G = (V, E, C))
2. IntuitionDualEngine (System 1 O(1) Hopfield Intuition vs System 2 Reasoning)
3. ScarConsolidator (Scar Tensor permanent consolidation into Attractor Memory Bank)
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from core.memory.causal_graph_extractor import CausalGraphExtractor
from core.memory.intuition_dual_engine import IntuitionDualEngine


class VisualCausalCognitionPipeline:
    """
    End-to-End Visual Causal Cognition Pipeline.

    Processes 2D frames (frame_t0, frame_t1), extracts kinematic topology graphs,
    and runs System 1 O(1) intuition or System 2 reasoning with Scar Tensor consolidation.
    """

    def __init__(
        self,
        spatial_dim: Tuple[int, int] = (64, 64),
        max_nodes: int = 8,
        attr_dim: int = 128,
        num_attractors: int = 512,
        friction_threshold: float = 0.25,
        random_seed: int = 42,
    ):
        self.spatial_dim = spatial_dim
        self.max_nodes = max_nodes
        self.attr_dim = attr_dim

        # 1. Causal Graph Extractor Module
        self.graph_extractor = CausalGraphExtractor(
            spatial_dim=spatial_dim,
            max_nodes=max_nodes,
        )

        # Projection matrix from raw causal vector (max_nodes * 2) to attr_dim
        np.random.seed(random_seed)
        input_dim = max_nodes * 2
        self.proj_w = np.random.randn(input_dim, attr_dim).astype(np.float32) * 0.1

        # 2. Intuition Dual Engine & Scar Consolidator
        self.intuition_engine = IntuitionDualEngine(
            dim=attr_dim,
            num_attractors=num_attractors,
            beta=8.0,
            friction_threshold=friction_threshold,
            random_seed=random_seed,
        )

    def extract_causal_graph(
        self, frame_t0: np.ndarray, frame_t1: np.ndarray
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Extracts causal graph G=(V,E,C) and projects topological features to attr_dim vector.
        """
        extractor_res = self.graph_extractor.forward(frame_t0, frame_t1)
        causal_vector_raw = extractor_res["causal_vector"]

        # Project to attr_dim
        if causal_vector_raw.shape[0] != self.proj_w.shape[0]:
            # Resize / pad if needed
            if causal_vector_raw.shape[0] < self.proj_w.shape[0]:
                causal_vector_raw = np.pad(
                    causal_vector_raw, (0, self.proj_w.shape[0] - causal_vector_raw.shape[0])
                )
            else:
                causal_vector_raw = causal_vector_raw[: self.proj_w.shape[0]]

        x_causal = np.dot(causal_vector_raw, self.proj_w)
        return extractor_res, x_causal

    def process_frames(
        self, frame_t0: np.ndarray, frame_t1: np.ndarray
    ) -> Dict[str, Any]:
        """
        Executes end-to-end pipeline:
        Frame Pair -> Causal Graph -> Intuition/Reasoning -> Scar Consolidation.

        Args:
            frame_t0: Frame t
            frame_t1: Frame t+1

        Returns:
            Dict containing pipeline results (predicted state, mode, friction, graph, etc.)
        """
        # Step 1: Extract Causal Graph & Feature Vector
        graph_res, x_causal = self.extract_causal_graph(frame_t0, frame_t1)

        # Step 2: Pass through Dual-Loop Intuition Engine
        cognition_res = self.intuition_engine.forward(x_causal)

        # Step 3: Combine outputs
        output = {
            "predicted_state": cognition_res["predicted_state"],
            "mode": cognition_res["mode"],
            "friction": cognition_res["friction"],
            "target_attractor_idx": cognition_res["target_attractor_idx"],
            "causal_graph": graph_res["causal_graph"],
            "node_masks": graph_res["node_masks"],
            "node_centers": graph_res["node_centers"],
            "causal_edges": graph_res["causal_edges"],
            "pivot_points": graph_res["pivot_points"],
        }
        return output
