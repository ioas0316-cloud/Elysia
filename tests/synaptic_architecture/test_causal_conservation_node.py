import pytest
import torch
from synaptic_architecture.causal_conservation_node import (
    BoundTensionField,
    TopologicalBoundarySkeleton,
    GenerativeRuleEngine,
    CausalConservationNode,
)


def test_bound_tension_field():
    dim = 32
    field = BoundTensionField(dim=dim)
    state = torch.randn(2, dim)
    context = torch.randn(2, dim)

    v_potential = field.compute_potential(state, context)

    assert v_potential.shape == (2,)
    assert torch.all(v_potential >= 0.0)


def test_topological_boundary_skeleton():
    dim = 32
    num_axioms = 6
    skeleton = TopologicalBoundarySkeleton(dim=dim, num_axioms=num_axioms)

    embeddings, i_c = skeleton.observe_at_scale(c_lens_scale=1.0)
    assert embeddings.shape == (num_axioms, dim)
    assert i_c.shape == (num_axioms,)

    # Scale invariance test
    embeddings_scaled, i_c_scaled = skeleton.observe_at_scale(c_lens_scale=10.0)
    assert embeddings_scaled.shape == (num_axioms, dim)
    # Structural invariant spectrum I_c remains preserved regardless of scale
    assert torch.allclose(i_c, i_c_scaled)


def test_generative_rule_engine():
    dim = 32
    num_rules = 4
    engine = GenerativeRuleEngine(dim=dim, num_rules=num_rules)
    state = torch.randn(2, dim)

    constraints = engine.generate_boundary_constraints(state)

    assert constraints.shape == (2, num_rules, dim)
    # Constraints are bound with ReLU (>= 0)
    assert torch.all(constraints >= 0.0)


def test_cc_node_forward_and_unfold():
    dim = 32
    node = CausalConservationNode(node_id="CC_NODE_TEST", dim=dim)

    batch_size = 2
    context = torch.randn(batch_size, dim)

    v_potential, v_react, i_c, constraints = node(context, c_lens_scale=1.0)

    assert v_potential.shape == (batch_size,)
    assert v_react.shape == (batch_size, dim)
    assert i_c.ndim == 1
    assert constraints.ndim == 3

    # Unfold mechanic
    steps = 4
    trajectory = node.unfold(context, steps=steps)
    assert trajectory.shape == (batch_size, steps + 1, dim)


def test_cc_node_tension_resistance():
    dim = 32
    node = CausalConservationNode(node_id="CC_NODE_TEST", dim=dim)

    # Moderate vs extreme noise
    batch_size = 1
    moderate_noise = torch.randn(batch_size, dim) * 0.1
    extreme_noise = torch.randn(batch_size, dim) * 10.0

    _, tension_mod = node.compute_tension_resistance(moderate_noise)
    _, tension_ext = node.compute_tension_resistance(extreme_noise)

    assert tension_ext > tension_mod


def test_cc_node_reversible_implosion():
    dim = 32
    node = CausalConservationNode(node_id="CC_NODE_TEST", dim=dim)

    original_state = node.latent_state.data.clone()

    # Implode and seal
    vault = node.implode_and_seal()
    assert node.is_sealed is True

    # Mutate state during active inference
    node.latent_state.data.add_(10.0)
    assert not torch.allclose(node.latent_state.data, original_state)

    # Reversibly recover from vault
    success = node.unseal_and_recover()
    assert success is True
    assert node.is_sealed is False
    assert torch.allclose(node.latent_state.data, original_state)
