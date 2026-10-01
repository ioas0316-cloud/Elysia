import pytest
import torch
import numpy as np

from synaptic_architecture.constructal_slaved_engine import (
    ConstructalSlavedModule,
    RealtimeSFALayer,
    PredictiveCodingLayer,
    NicheConstructionMemory,
    MetaLens,
    GenerativeSelfModel,
    SecondOrderCyberneticsEngine,
    ActiveInferenceNicheEngine,
)


def test_constructal_slaved_module_bifurcation():
    torch.manual_seed(42)
    in_dim, out_dim = 16, 16
    module = ConstructalSlavedModule(in_dim, out_dim, num_order_params=4, split_threshold=0.5)

    # Initial input (low stress)
    x = torch.randn(8, in_dim)
    out1 = module(x)
    assert out1.shape == (8, out_dim)

    # Force artificial high stress to trigger dynamic branching
    module.resistance_ema = torch.tensor([100.0])
    out2 = module(x)
    assert out2.shape == (8, out_dim)
    assert module.branch_path is not None  # Branch dynamically created

    # Test categorical lens optic methods
    pred = module.get_prediction(x)
    assert pred.shape == (8, out_dim)
    corrected = module.put_error(x, torch.randn(8, out_dim))
    assert corrected.shape == (8, in_dim)


def test_realtime_sfa_layer_slowness():
    torch.manual_seed(42)
    np.random.seed(42)
    seq_len, in_features = 1000, 10
    t = np.linspace(0, 10, seq_len)

    # Slow latent wave + fast noise
    slow_wave = np.sin(0.5 * t)
    fast_noise = np.random.randn(seq_len, in_features) * 1.5

    x_data = np.zeros((seq_len, in_features))
    for i in range(in_features):
        x_data[:, i] = slow_wave + fast_noise[:, i]

    x_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(0)  # [1, 1000, 10]

    sfa = RealtimeSFALayer(in_features=10, num_slow_features=1)
    q_extracted = sfa(x_tensor, is_training_stream=True).squeeze().detach().numpy()

    # Calculate slowness (mean square derivative normalized by variance)
    def slowness(signal):
        diff = np.diff(signal)
        return np.mean(diff ** 2) / (np.var(signal) + 1e-8)

    slowness_input = np.mean([slowness(x_data[:, i]) for i in range(in_features)])
    slowness_extracted = slowness(q_extracted)

    # Extracted feature should be significantly slower than noisy raw input
    assert slowness_extracted < slowness_input


def test_predictive_coding_layer_relaxation():
    torch.manual_seed(42)
    dim_current, dim_higher = 12, 6
    pc_layer = PredictiveCodingLayer(dim_current, dim_higher, lr_mu=0.1)

    x_sensory = torch.randn(4, dim_current)
    mu_prior = torch.randn(4, dim_higher)

    # Forward prediction before relaxation
    x_pred_init = pc_layer.get_prediction(mu_prior)
    err_init = torch.norm(x_sensory - x_pred_init).item()

    # Relax states
    mu_converged, final_error = pc_layer(x_sensory, mu_prior, steps=20)
    err_final = torch.norm(final_error).item()

    # Free energy error should decrease after state relaxation
    assert err_final <= err_init


def test_niche_construction_memory():
    torch.manual_seed(42)
    memory = NicheConstructionMemory(memory_slots=20, dim=16, decay=0.01, lr_niche=0.2)

    query_state = torch.randn(2, 16)
    context_init = memory.read_niche_context(query_state)
    assert context_init.shape == (2, 16)

    # Perform action and construct niche
    action = torch.randn(2, 16)
    memory.construct_niche(query_state, action)

    # Context after constructing niche well
    context_after = memory.read_niche_context(query_state)
    assert context_after.shape == (2, 16)


def test_second_order_cybernetics_engine():
    torch.manual_seed(42)
    dim_sensory, dim_world, dim_self = 8, 4, 2
    engine = SecondOrderCyberneticsEngine(dim_sensory, dim_world, dim_self)

    # Familiar input frame
    x_familiar = torch.randn(1, dim_sensory) * 0.1
    res1 = engine.step_environment(x_familiar, relaxation_steps=5)

    # Novel / Out-of-Distribution input frame (Domain Shift)
    x_novel = torch.randn(1, dim_sensory) * 5.0 + 3.0
    res2 = engine.step_environment(x_novel, relaxation_steps=5)

    # Epistemic uncertainty should rise or adapt under novel domain shift
    assert "epistemic_uncertainty" in res2
    assert "free_energy" in res2
    assert res2["mu_world"].shape == (1, dim_world)
    assert res2["mu_self"].shape == (1, dim_self)


def test_active_inference_niche_engine_integration():
    torch.manual_seed(42)
    engine = ActiveInferenceNicheEngine(dim_sensory=8, dim_higher=16, memory_slots=20)

    x_input = torch.randn(2, 8)
    outputs = engine(x_input)

    assert "x_slaved" in outputs
    assert "q_order" in outputs
    assert "mu_converged" in outputs
    assert "pc_error" in outputs
    assert "cybernetics" in outputs
    assert outputs["x_slaved"].shape == (2, 8)
