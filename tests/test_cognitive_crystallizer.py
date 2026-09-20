"""
Unit tests for Cognitive Crystallizer Engine backend module.
"""

import math
import pytest
import torch
from core.physics.cognitive_crystallizer import CognitiveCrystallizerEngine


def test_crystallizer_initialization():
    engine = CognitiveCrystallizerEngine(num_nodes=50, phi_solid=5.0, phi_gas=0.2)
    assert engine.num_nodes == 50
    assert engine.phi_solid == 5.0
    assert engine.phi_gas == 0.2


def test_forward_phase_transition():
    engine = CognitiveCrystallizerEngine(num_nodes=20, phi_solid=3.0, phi_gas=0.2)
    X = torch.randn(20, 3)
    V = torch.randn(20, 3)

    out = engine(X, V, entropy=1.0)

    assert "Phi" in out
    assert "A" in out
    assert "MetricTensor" in out
    assert "Gammas" in out
    assert "phase_ratios" in out

    # Check metric tensor dimensions (20, 3, 3)
    assert out["MetricTensor"].shape == (20, 3, 3)
    # Check time dilation gammas dimensions (20,)
    assert out["Gammas"].shape == (20,)
    # Verify gammas >= 1.0
    assert torch.all(out["Gammas"] >= 1.0)


test_forward_phase_transition()


def test_topological_shear():
    engine = CognitiveCrystallizerEngine(num_nodes=30)
    X = torch.zeros(30, 3)
    # Place node 0 at origin, node 1 close, node 29 far away
    X[1] = torch.tensor([0.2, 0.0, 0.0])
    X[29] = torch.tensor([10.0, 10.0, 10.0])
    V = torch.zeros(30, 3)

    impact_point = torch.tensor([0.0, 0.0, 0.0])
    impact_vector = torch.tensor([5.0, 0.0, 0.0])

    X_new, V_new, mask = engine.apply_topological_shear(X, V, impact_point, impact_vector, radius=1.0)

    # Nodes near origin should have high velocity change
    assert mask[0] == 1.0
    assert mask[1] == 1.0
    assert mask[29] == 0.0
    assert torch.norm(V_new[0]) > 0.0


def test_macro_node_contraction():
    engine = CognitiveCrystallizerEngine(num_nodes=10, phi_solid=1.0)
    X = torch.rand(10, 3)
    V = torch.rand(10, 3)
    A = torch.ones(10, 10)  # All connected as solid

    X_macro, V_macro, cluster_map = engine.contract_macro_nodes(X, V, A)

    # Since all nodes are connected in solid phase, should contract to 1 macro-node
    assert X_macro.size(0) == 1
    assert V_macro.size(0) == 1
    assert torch.all(cluster_map == 0)


if __name__ == "__main__":
    pytest.main([__file__])
