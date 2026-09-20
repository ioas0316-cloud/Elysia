"""
Unit tests for Phase 3: Spatial Mirror & Dynamic Metric Field
"""

import pytest
import numpy as np
from synaptic_architecture.causal_receptor import CausalReceptor
from synaptic_architecture.spatial_mirror import SpatialMirror


def test_spatial_mirror_mapping_and_propagation():
    receptor = CausalReceptor()
    graph = receptor.deconstruct_code("a = x + 10\nb = a * 2")

    mirror = SpatialMirror(num_nodes=32)
    mirror.map_causal_graph(graph)

    # Inject external phase signal
    ext_phase = 1.5708  # pi / 2
    phases_after = mirror.inject_phase_signal(target_id=0, ext_phase=ext_phase, coupling_k=0.3)

    assert phases_after.shape[0] == 32
    # Target node phase should move towards external phase
    assert np.abs(phases_after[0] - ext_phase) < 1.5708


def test_self_healing_rerouting():
    mirror = SpatialMirror(num_nodes=16)
    damaged_id = 2
    alternate_id = mirror.reroute_damaged_node(damaged_id)

    assert alternate_id != damaged_id
    assert mirror.metric_matrix[damaged_id, 0] == 100.0


if __name__ == "__main__":
    test_spatial_mirror_mapping_and_propagation()
    test_self_healing_rerouting()
    print("All SpatialMirror tests passed!")
