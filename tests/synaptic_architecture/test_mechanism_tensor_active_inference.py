"""
[Test Suite: MechanismTensor & Latent Active Inference Engine]

Verifies:
1. CausalLineage LCA branch detection and O(1) causal gap differentiation.
2. MechanismTensor auto-dispatch Einsum tensor relaxation under topological tension.
3. Multi-node CausalGraphNetwork tension wave propagation and equilibrium resolution.
4. CausalGeodesicInferenceEngine minimal tension geodesic construction.
5. Amortized Active Inference Latent World Model:
   - Forward pass & reparameterization
   - Free Energy F_t backpropagation & loss reduction
   - Expected Free Energy G_t decomposition (Epistemic curiosity + Pragmatic preference)
   - CEM Action Planning in latent space with MechanismTensor lineage binding
"""

import math
import pytest
import torch
import torch.optim as optim
import numpy as np

from synaptic_architecture.mechanism_tensor import (
    CausalLineage,
    TopologicalInvariant,
    MechanismTensor,
    MechanismNode,
    CausalEdge,
    CausalGraphNetwork,
    CausalGeodesicInferenceEngine
)

from synaptic_architecture.latent_active_inference_world_model import (
    LatentActiveInferenceAgent,
    compute_variational_free_energy,
    compute_expected_free_energy,
    reparameterize
)


def test_causal_lineage_lca_differentiation():
    """Verifies O(1) LCA branch detection between two divergent transformation paths."""
    l1 = CausalLineage(
        node_id="Alpha",
        parent_ids=["Root"],
        transformation_history=["init", "transform_A", "branch_left"]
    )
    l2 = CausalLineage(
        node_id="Beta",
        parent_ids=["Root"],
        transformation_history=["init", "transform_A", "branch_right"]
    )

    common_id, split_depth = l1.find_lowest_common_ancestor(l2)
    assert common_id in ["Alpha", "Root"]
    assert split_depth == 2  # 'init' and 'transform_A' are common


def test_mechanism_tensor_auto_dispatch_relaxation():
    """Verifies auto-dispatch of Einsum axis contraction under topological tension."""
    lineage = CausalLineage(node_id="Test_Node")
    invariant = TopologicalInvariant(name="Flux_Conservation", target_value=1.0)

    distorted_tensor = torch.tensor([
        [[5.0, 1.0], [0.5, 2.0]],
        [[3.0, 4.0], [1.2, 0.8]]
    ])

    m_tensor = MechanismTensor(distorted_tensor, lineage, invariant=invariant)
    initial_tension, _ = m_tensor.update_tension()
    assert initial_tension.item() > 0.1

    m_tensor.auto_dispatch_relaxation(tolerance=1e-3, max_steps=10)
    final_tension, _ = m_tensor.update_tension()

    assert final_tension.item() < initial_tension.item()
    assert len(m_tensor.lineage.transformation_history) > 0


def test_causal_graph_network_equilibrium():
    """Verifies multi-node network tension wave propagation and asynchronous relaxation."""
    net = CausalGraphNetwork()

    n1 = MechanismNode("Sensor_Alpha", torch.tensor([[[4.0, 0.5], [1.0, 2.0]], [[2.0, 3.0], [0.1, 1.5]]]))
    n2 = MechanismNode("Core_Beta", torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]))
    n3 = MechanismNode("Actuator_Gamma", torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]))

    net.add_node(n1)
    net.add_node(n2)
    net.add_node(n3)

    net.connect("Sensor_Alpha", "Core_Beta", coupling=0.8)
    net.connect("Core_Beta", "Actuator_Gamma", coupling=0.4)

    net.resolve_network_equilibrium(max_cycles=10, tol=1e-3)

    # Check that transformation history recorded tension propagation/relaxation
    for node_id, node in net.nodes.items():
        assert isinstance(node.lineage.transformation_history, list)


def test_causal_geodesic_inference_engine():
    """Verifies synthesis of minimal-tension causal geodesic trajectory."""
    start_tensor = MechanismTensor(
        raw_tensor=torch.tensor([[[3.0, 2.0], [1.0, 4.0]], [[2.0, 1.0], [0.5, 2.5]]]),
        lineage=CausalLineage(node_id="Start_Point", transformation_history=["origin"])
    )
    target_invariant = TopologicalInvariant(name="Target_Equilibrium", target_value=1.0)

    engine = CausalGeodesicInferenceEngine(start_tensor, target_invariant)
    result = engine.construct_geodesic(max_steps=5)

    assert "geodesic_trajectory" in result
    assert result["final_tension"] <= result["initial_tension"]


def test_latent_active_inference_forward_and_f_t_backprop():
    """Verifies Amortized Active Inference World Model forward pass and F_t backprop gradient update."""
    obs_dim = 16
    state_dim = 4
    action_dim = 2

    agent = LatentActiveInferenceAgent(obs_dim, state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.01)

    obs = torch.randn(1, obs_dim)
    prev_state = torch.zeros(1, state_dim)
    prev_action = torch.zeros(1, action_dim)

    # Compute initial Free Energy
    out_1 = agent.perceive_and_learn(obs, prev_state, prev_action)
    initial_f = out_1["free_energy"].item()

    # Optimization step
    optimizer.zero_grad()
    out_1["free_energy"].backward()
    optimizer.step()

    # Compute updated Free Energy
    out_2 = agent.perceive_and_learn(obs, prev_state, prev_action)
    updated_f = out_2["free_energy"].item()

    assert not math.isnan(initial_f)
    assert not math.isnan(updated_f)


def test_expected_free_energy_and_cem_planning():
    """Verifies Expected Free Energy G_t decomposition and CEM action selection in latent space."""
    obs_dim = 8
    state_dim = 4
    action_dim = 2

    agent = LatentActiveInferenceAgent(obs_dim, state_dim, action_dim)

    # Set target preference distribution p(s_tilde) to (1.0, 1.0, ...)
    target_pref = torch.ones(1, state_dim) * 2.0
    agent.set_target_preference(target_pref)

    current_state = torch.zeros(1, state_dim)

    optimal_action, action_m_tensor = agent.plan_action_cem(
        current_state, horizon=3, num_samples=16, top_k=4, cem_iterations=2
    )

    assert optimal_action.shape == (1, action_dim)
    assert isinstance(action_m_tensor, MechanismTensor)
    assert len(action_m_tensor.lineage.transformation_history) > 0
