"""
Unit Tests for Internal Metric Field Phase-Lock Engine (PyTorch Module)
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

from core.physics.phase_lock_engine import PhaseLockEngine


def test_phase_lock_engine_initialization():
    engine = PhaseLockEngine(num_nodes=10, feature_dim=3, phi_solid=5.0, phi_gas=0.2)
    assert engine.num_nodes == 10
    assert engine.C.shape == (10, 10)
    assert engine.M.shape == (10, 10)
    assert (engine.C == 1.0).all()
    assert (engine.M == 0.0).all()


def test_dense_phase_transitions():
    num_nodes = 5
    engine = PhaseLockEngine(num_nodes=num_nodes, phi_solid=5.0, phi_gas=0.2)

    # 1. Stationary lattice nodes -> Mobility low, Constraint high -> Solid Phase (Phi >= 5.0)
    X_solid = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0]
    ])
    V_solid = torch.zeros(num_nodes, 3)

    Phi_s, A_s = engine(X_solid, V_solid)
    assert Phi_s.shape == (num_nodes, num_nodes)
    assert A_s.shape == (num_nodes, num_nodes)
    # Off-diagonal elements should have high Phi (Solid phase >= phi_solid)
    assert Phi_s[0, 1] >= engine.phi_solid
    assert A_s[0, 1] == 1.0

    # 2. Inject Moderate Kinetic Energy -> Liquid Phase (0.2 <= Phi < 5.0)
    X_liquid = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0]
    ])
    V_liquid = torch.tensor([
        [2.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0]
    ])

    # Run multiple steps to let mobility build up and decay constraint
    for _ in range(5):
        Phi_l, A_l = engine(X_liquid, V_liquid)

    dist = engine.get_phase_distribution(Phi_l)
    assert "solid_ratio" in dist
    assert "liquid_ratio" in dist
    assert "gas_ratio" in dist

    # 3. Inject Extreme Velocity Shock -> Gas Phase (Phi < 0.2)
    X_gas = torch.randn(num_nodes, 3) * 10.0
    V_gas = torch.randn(num_nodes, 3) * 100.0

    for _ in range(10):
        Phi_g, A_g = engine(X_gas, V_gas)

    dist_g = engine.get_phase_distribution(Phi_g)
    assert dist_g["gas_ratio"] > 0.0 or dist_g["liquid_ratio"] > 0.0


def test_sparse_csr_phase_lock_update():
    num_nodes = 4
    engine = PhaseLockEngine(num_nodes=num_nodes, r_cut=3.0)

    X = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [10.0, 0.0, 0.0]  # Far away node (> r_cut)
    ], dtype=torch.float32)

    V = torch.zeros(num_nodes, 3, dtype=torch.float32)

    # CSR edges: 0->1, 0->2, 1->2, 2->3 (3 is far away)
    row_ptr = torch.tensor([0, 2, 3, 4, 4], dtype=torch.int32)
    col_idx = torch.tensor([1, 2, 2, 3], dtype=torch.int32)
    num_edges = col_idx.size(0)

    C_edge = torch.ones(num_edges, dtype=torch.float32)
    M_edge = torch.zeros(num_edges, dtype=torch.float32)

    C_out, M_out, Phi_edge, A_edge = engine.forward_sparse(X, V, row_ptr, col_idx, C_edge, M_edge)

    assert Phi_edge.shape == (num_edges,)
    assert A_edge.shape == (num_edges,)
    # Edge 2->3 is far away (> r_cut=3.0), so Phi and A should be 0
    assert Phi_edge[3].item() == 0.0
    assert A_edge[3].item() == 0.0


def test_macro_node_early_contraction():
    num_nodes = 6
    engine = PhaseLockEngine(num_nodes=num_nodes)

    X = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],  # Cluster 1 (0, 1)
        [5.0, 0.0, 0.0],
        [5.1, 0.0, 0.0],  # Cluster 2 (2, 3)
        [10.0, 0.0, 0.0],
        [10.1, 0.0, 0.0]  # Cluster 3 (4, 5)
    ], dtype=torch.float32)

    V = torch.zeros(num_nodes, 3, dtype=torch.float32)

    # Adjacency matrix with solid connections within pairs
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    A[0, 1] = A[1, 0] = 1.0
    A[2, 3] = A[3, 2] = 1.0
    A[4, 5] = A[5, 4] = 1.0

    X_active, V_active, cluster_map = engine.contract_solid_clusters(X, V, A)

    # 6 micro-nodes grouped into 3 macro-nodes
    assert X_active.shape[0] == 3
    assert V_active.shape[0] == 3
    assert cluster_map.shape[0] == num_nodes
    assert cluster_map[0] == cluster_map[1]
    assert cluster_map[2] == cluster_map[3]
    assert cluster_map[4] == cluster_map[5]


def test_reset_and_resize():
    engine = PhaseLockEngine(num_nodes=4)
    assert engine.C.shape == (4, 4)

    # Automatically resizes state buffers when a batch with a different number of nodes is passed
    X_new = torch.randn(8, 3)
    V_new = torch.randn(8, 3)
    Phi, A = engine(X_new, V_new)

    assert Phi.shape == (8, 8)
    assert engine.num_nodes == 8
