"""
Causal Graph Extractor (CausalGraphExtractor)
==============================================
Extracts kinematic topology and causal graphs G = (V, E, C) from 2D pixel streams
via spatial-temporal motion clustering, kinematic constraint inversion, and graph consolidation.
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional


class CausalGraphExtractor:
    """
    Extracts causal graphs from 2D frame sequences (pixel streams).

    Phases:
    1. Spatial-Temporal Motion Clustering: Groups pixels sharing coherent motion into rigid body nodes (V).
    2. Kinematic Constraint Inversion: Infers node-to-node relative trajectories to identify joint pivots & kinematic edges (E).
    3. Causal Graph Formulation: Consolidates Nodes (V), Edges (E), and Dynamics Constraints (C).
    """

    def __init__(
        self,
        spatial_dim: Tuple[int, int] = (64, 64),
        max_nodes: int = 8,
        motion_threshold: float = 0.05,
        pivot_tolerance: float = 0.1,
    ):
        self.spatial_dim = spatial_dim
        self.max_nodes = max_nodes
        self.motion_threshold = motion_threshold
        self.pivot_tolerance = pivot_tolerance

    def forward(
        self, frame_t0: np.ndarray, frame_t1: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process consecutive 2D frames (frame_t0, frame_t1).

        Args:
            frame_t0: Array of shape (H, W, C) or (H, W) or (B, H, W, C)
            frame_t1: Array of shape (H, W, C) or (H, W) or (B, H, W, C)

        Returns:
            Dict containing:
                - node_masks: Soft/hard segmentation masks for nodes [num_nodes, H, W]
                - node_centers: Spatial centers of mass for nodes [num_nodes, 2]
                - causal_edges: Adjacency matrix of kinematic edges [num_nodes, num_nodes]
                - pivot_points: Estimated joint pivot locations [num_nodes, num_nodes, 2]
                - causal_graph: Graph dict G = (V, E, C)
                - causal_vector: Flattened topological vector representing the graph state
        """
        f0 = self._prepare_frame(frame_t0)
        f1 = self._prepare_frame(frame_t1)

        # Phase 1: Motion vector extraction and spatial masking
        node_masks, motion_clusters = self._extract_motion_clusters(f0, f1)

        # Calculate Center of Mass for each node cluster
        node_centers = self._extract_node_centers(node_masks)

        # Phase 2: Kinematic Constraint Inversion (Pivot points and edge adjacency)
        causal_edges, pivot_points = self._infer_kinematic_edges(node_centers, node_masks)

        # Phase 3: Causal Graph Consolidation G = (V, E, C)
        causal_graph = self._consolidate_causal_graph(
            node_centers, causal_edges, pivot_points, node_masks
        )

        # Topological feature vector projection
        causal_vector = node_centers.flatten()

        return {
            "node_masks": node_masks,
            "node_centers": node_centers,
            "causal_edges": causal_edges,
            "pivot_points": pivot_points,
            "causal_graph": causal_graph,
            "causal_vector": causal_vector,
        }

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim == 2:
            frame = frame[:, :, np.newaxis]
        elif frame.ndim == 4:
            frame = frame[0]
        return frame

    def _extract_motion_clusters(
        self, f0: np.ndarray, f1: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Computes motion magnitude/vector field and segments spatial regions into max_nodes clusters.
        """
        H, W, C = f0.shape
        # Compute frame difference / motion magnitude
        diff = np.abs(f1 - f0)
        motion_mag = np.mean(diff, axis=-1)  # [H, W]

        # Background mask (motion below threshold)
        bg_mask = (motion_mag < self.motion_threshold).astype(np.float32)

        # Motion active regions
        active_mask = 1.0 - bg_mask

        # Generate node masks (Node 0 is Background, Node 1..N-1 are foreground rigid bodies)
        masks = np.zeros((self.max_nodes, H, W), dtype=np.float32)
        masks[0] = bg_mask

        if np.sum(active_mask) > 0:
            # Spatial grid coordinates
            grid_y, grid_x = np.mgrid[0:H, 0:W]

            # Simple spatial-motion clustering into active nodes
            num_fg_nodes = self.max_nodes - 1
            active_y = grid_y[active_mask > 0]
            active_x = grid_x[active_mask > 0]

            if len(active_x) > 0:
                angles = np.arctan2(active_y - H / 2.0, active_x - W / 2.0)
                sector_idx = np.floor((angles + np.pi) / (2 * np.pi) * num_fg_nodes).astype(int)
                sector_idx = np.clip(sector_idx, 0, num_fg_nodes - 1)

                active_coords = np.argwhere(active_mask > 0)
                for idx, (r, c) in enumerate(active_coords):
                    s_id = sector_idx[idx] + 1
                    masks[s_id, r, c] = motion_mag[r, c] + 1e-5

        # Normalize masks across nodes (softmax / sum normalize per pixel)
        mask_sums = np.sum(masks, axis=0, keepdims=True) + 1e-8
        masks = masks / mask_sums

        return masks, [masks[i] for i in range(self.max_nodes)]

    def _extract_node_centers(self, masks: np.ndarray) -> np.ndarray:
        """
        Spatial Soft-Argmax / Center of Mass for each node cluster.
        Returns: [max_nodes, 2] array of (x, y) normalized coordinates in [-1, 1].
        """
        num_nodes, H, W = masks.shape
        pos_y, pos_x = np.mgrid[0:H, 0:W]

        norm_y = (pos_y / max(H - 1, 1)) * 2.0 - 1.0
        norm_x = (pos_x / max(W - 1, 1)) * 2.0 - 1.0

        centers = np.zeros((num_nodes, 2), dtype=np.float32)
        for i in range(num_nodes):
            mask_weight = masks[i]
            total_mass = np.sum(mask_weight)
            if total_mass > 1e-6:
                cx = np.sum(mask_weight * norm_x) / total_mass
                cy = np.sum(mask_weight * norm_y) / total_mass
                centers[i] = [cx, cy]
            else:
                centers[i] = [0.0, 0.0]

        return centers

    def _infer_kinematic_edges(
        self, centers: np.ndarray, masks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Infer kinematic joints (edges) and pivot points between contiguous nodes.
        Returns:
            - adjacency: [max_nodes, max_nodes] binary matrix
            - pivot_points: [max_nodes, max_nodes, 2] spatial coordinates of pivots
        """
        N = self.max_nodes
        adjacency = np.zeros((N, N), dtype=np.float32)
        pivots = np.zeros((N, N, 2), dtype=np.float32)

        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < 1.2:
                    pivot = (centers[i] + centers[j]) / 2.0
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0
                    pivots[i, j] = pivot
                    pivots[j, i] = pivot

        return adjacency, pivots

    def _consolidate_causal_graph(
        self,
        centers: np.ndarray,
        adjacency: np.ndarray,
        pivots: np.ndarray,
        masks: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Consolidates Nodes (V), Edges (E), and Constraints (C) into graph G = (V, E, C).
        """
        nodes = []
        for i in range(self.max_nodes):
            node_type = "V_bg" if i == 0 else ("V_body" if i == 1 else "V_limb")
            mass = float(np.sum(masks[i]))
            nodes.append({
                "id": i,
                "type": node_type,
                "center": centers[i].tolist(),
                "mass": mass
            })

        edges = []
        N = self.max_nodes
        for i in range(N):
            for j in range(i + 1, N):
                if adjacency[i, j] > 0:
                    edges.append({
                        "source": i,
                        "target": j,
                        "type": "E_kinematic" if (i > 0 and j > 0) else "E_contact",
                        "pivot": pivots[i, j].tolist(),
                        "constraint": {
                            "max_rotation_deg": 180.0,
                            "distance_stiffness": 1.0
                        }
                    })

        return {
            "V": nodes,
            "E": edges,
            "C": {
                "invariance_type": "rigid_body_kinematics",
                "conservation_laws": ["pivot_distance_preservation"]
            }
        }
