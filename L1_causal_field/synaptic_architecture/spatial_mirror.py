"""
Spatial Mirror & Dynamic Metric Field (Phase 3)
==============================================
Maps topological causal interaction densities d(i, j) 1:1 into physical/topological
memory layouts. Performs localized phase synchronization and dynamic metric field deformation.
Supports self-healing rerouting when nodes or memory sub-regions suffer damage.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from synaptic_architecture.causal_receptor import AtomicCausalGraph


class SpatialMirror:
    """Dynamic Metric Field and Spatial Memory Topology Mapper."""

    def __init__(self, num_nodes: int = 256):
        self.num_nodes = num_nodes
        self.phases = np.zeros(num_nodes, dtype=np.float32)
        self.energies = np.ones(num_nodes, dtype=np.float32)
        # Default metric distance matrix: all initialized to topological default 2.0
        self.metric_matrix = np.full((num_nodes, num_nodes), fill_value=2.0, dtype=np.float32)
        np.fill_diagonal(self.metric_matrix, 0.0)

        # Try loading native C++ bridge if compiled
        self.cpp_bridge = None
        try:
            import elysia_cpp_bridge
            self.cpp_bridge = elysia_cpp_bridge.CppElysiaBridge(num_nodes)
        except ImportError:
            self.cpp_bridge = None

    def map_causal_graph(self, graph: AtomicCausalGraph) -> None:
        """Maps causal graph distances 1:1 onto spatial metric memory layout."""
        graph_dist = graph.compute_causal_distance_matrix()
        g_len = min(graph_dist.shape[0], self.num_nodes)

        # Map graph distances into upper-left block of metric matrix
        self.metric_matrix[:g_len, :g_len] = graph_dist[:g_len, :g_len]

        if self.cpp_bridge:
            self.cpp_bridge.set_metric_matrix(self.metric_matrix)

    def inject_phase_signal(
        self,
        target_id: int,
        ext_phase: float,
        coupling_k: float = 0.2,
        deform_metric: bool = True
    ) -> np.ndarray:
        """Injects external phase into target node, propagates across metric field,

        and dynamically deforms metric topology based on phase differentials.
        """
        if target_id >= self.num_nodes:
            target_id = target_id % self.num_nodes

        if self.cpp_bridge:
            self.cpp_bridge.inject_external_phase(target_id, ext_phase, coupling_k)
            self.phases = self.cpp_bridge.get_phases()
            return self.phases

        # NumPy Vectorized Fallback
        # 1. Target node phase update (Kuramoto phase torque)
        phase_diff = ext_phase - self.phases[target_id]
        self.phases[target_id] += coupling_k * np.sin(phase_diff)
        self.phases[target_id] = np.fmod(self.phases[target_id] + 2.0 * np.pi, 2.0 * np.pi)

        # 2. Spatial propagation via metric distance attenuation exp(-d(i, j))
        distances = self.metric_matrix[target_id, :]
        spatial_influence = np.exp(-distances)

        neighbor_diffs = self.phases[target_id] - self.phases
        self.phases += coupling_k * spatial_influence * np.sin(neighbor_diffs)
        self.phases = np.fmod(self.phases + 2.0 * np.pi, 2.0 * np.pi)

        # 3. Dynamic Metric Field Deformation (Phase-Locking Step)
        if deform_metric:
            # Shift distances to minimize phase error over time
            phase_errors = np.abs(self.phases[target_id] - self.phases)
            # Deformation torque: enlarge distance if phase diff is high, shorten if synced
            self.metric_matrix[target_id, :] += 0.01 * np.sin(phase_errors)
            self.metric_matrix[:, target_id] = self.metric_matrix[target_id, :]
            # Clamp minimum spatial threshold
            np.clip(self.metric_matrix, a_min=0.1, a_max=20.0, out=self.metric_matrix)
            np.fill_diagonal(self.metric_matrix, 0.0)

        return self.phases

    def reroute_damaged_node(self, damaged_node_id: int) -> int:
        """Self-healing spatial re-routing: Isolates a physically damaged node

        and shifts causal flow to the nearest healthy topological neighbor.
        """
        if damaged_node_id >= self.num_nodes:
            return 0

        # Isolate damaged node in metric space
        self.metric_matrix[damaged_node_id, :] = 100.0
        self.metric_matrix[:, damaged_node_id] = 100.0

        # Find healthy neighbor with lowest average distance
        avg_dists = np.mean(self.metric_matrix, axis=1)
        avg_dists[damaged_node_id] = np.inf
        alternate_node_id = int(np.argmin(avg_dists))

        return alternate_node_id
