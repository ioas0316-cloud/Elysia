import torch
import pytest
from synaptic_architecture.predictive_coding import PredictiveCodingNet


def test_predictive_coding_free_energy_reduction():
    torch.manual_seed(42)
    layer_dims = [64, 32, 16]
    pc_net = PredictiveCodingNet(layer_dims, lr_r=0.08, lr_w=0.02)

    x = torch.randn(16, 64)

    # Initial Free Energy
    states, errors, pre_acts = pc_net.relax_states(x, relaxation_steps=30)
    initial_fe = pc_net.compute_free_energy(errors)

    # Train over 15 iterations without loss.backward()
    for epoch in range(15):
        states, errors, pre_acts = pc_net.relax_states(x, relaxation_steps=30)
        pc_net.update_weights(states, errors, pre_acts)

    states, errors, pre_acts = pc_net.relax_states(x, relaxation_steps=30)
    final_fe = pc_net.compute_free_energy(errors)

    assert final_fe < initial_fe, f"Free energy should decrease: {initial_fe} -> {final_fe}"


def test_predictive_coding_dimension_preserving_errors():
    layer_dims = [64, 32, 16]
    pc_net = PredictiveCodingNet(layer_dims)
    x = torch.randn(8, 64)

    states, errors, pre_acts = pc_net.relax_states(x, relaxation_steps=10)

    # Verify error tensors preserve layer dimensions
    assert errors[0].shape == (8, 64)  # Layer 0 error \varepsilon_0 \in R^{D_0}
    assert errors[1].shape == (8, 32)  # Layer 1 error \varepsilon_1 \in R^{D_1}
