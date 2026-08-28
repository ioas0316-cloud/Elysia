import pytest
import numpy as np
from core.sensory.embodied_causal_substrate import EmbodiedCausalSubstrate, EngramSymbol
from synaptic_architecture.causal_membrane import CausalMembrane
from simulators.embodied_sandbox_sim import EmbodiedSandboxSimulator

def test_active_inference_top_down_noise_attenuation():
    """
    Benchmark Test 1: Active Inference Top-Down Projection & Noise Attenuation.
    Verifies that perpendicular background noise is suppressed (Phase Cancellation)
    when top-down intent field is active.
    """
    substrate = EmbodiedCausalSubstrate(vector_dim=8)
    intent = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    substrate.set_intent(intent)

    # Signal = Intent + High Perpendicular Noise
    raw_signal = np.array([1.0, 5.0, -4.0, 3.0, 2.0, -5.0, 6.0, 1.0])
    data_stream = [raw_signal]

    result = substrate.process_data_as_lens(data_stream, layer_name="C_meso")

    # Noise reduction ratio should be significant (> 50%)
    assert result["noise_reduction_ratio"] > 0.5, f"Noise reduction ratio too low: {result['noise_reduction_ratio']}"

def test_structural_data_lens_self_discovery():
    """
    Benchmark Test 2: Structural Data Lens Self-Discovery.
    Verifies that the substrate automatically discerns static spatial invariants (image-like)
    from spatiotemporal trajectories (video/flow-like) without artificial class labels.
    """
    substrate = EmbodiedCausalSubstrate(vector_dim=8)
    intent = np.ones(8)
    substrate.set_intent(intent)

    # 1. Static Spatial Snapshot (Identical frames across time)
    static_frame = np.ones(8) * 0.8
    static_stream = [static_frame for _ in range(5)]
    res_static = substrate.process_data_as_lens(static_stream)
    assert res_static["invariant_type"] == "spatial"
    assert res_static["temporal_variance"] < 0.05

    # 2. Dynamic Spatiotemporal Trajectory (Varying frames across time)
    dynamic_stream = [np.ones(8) * (i * 0.5) for i in range(5)]
    res_dynamic = substrate.process_data_as_lens(dynamic_stream)
    assert res_dynamic["invariant_type"] == "spatiotemporal"
    assert res_dynamic["temporal_variance"] >= 0.05

def test_reciprocal_boundary_dynamics_and_membrane_resistance():
    """
    Benchmark Test 3: Reciprocal Boundary Dynamics & Multi-Layered Membrane Resistance.
    Verifies Inside-Out vs Outside-In feedback loop across C_macro, C_meso, C_micro layers.
    """
    membrane = CausalMembrane(vector_dim=8)
    intent = np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    membrane.project_top_down_intent(intent)

    # High resistance environmental counter-impact
    raw_impact = np.array([-2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    filtered_signal, resonance, friction = membrane.interact_bottom_up_resistance(raw_impact, layer_name="C_meso")

    # Counter-impact should cause high friction and low resonance
    assert friction > 1.5
    assert resonance < 0.1
    # Check layer potential offset was adjusted dynamically in response to high friction
    assert np.linalg.norm(membrane.meso_layer.potential_offset) > 0.0

def test_spontaneous_symbol_transduction_and_engram():
    """
    Benchmark Test 4: Spontaneous Minimal Action Symbol Transduction & Engram Consolidation.
    Verifies that when friction converges below minimal threshold, an EngramSymbol is extracted.
    """
    substrate = EmbodiedCausalSubstrate(vector_dim=8)
    intent = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    substrate.set_intent(intent)

    # Low-friction matching stream
    matching_frame = np.array([1.1, 0.9, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    stream = [matching_frame for _ in range(3)]

    result = substrate.process_data_as_lens(stream)

    assert result["converged_engram"] is not None
    engram: EngramSymbol = result["converged_engram"]
    assert engram.invariant_type == "spatial"
    assert len(substrate.engram_bank) == 1
    assert "engram_1" in engram.symbol_id

def test_embodied_sandbox_simulator_integration():
    """
    Integration Test: Runs full embodied sandbox simulation and checks agent interaction.
    """
    sim = EmbodiedSandboxSimulator(grid_size=6, vector_dim=8)
    history = sim.run_full_trajectory(max_steps=10)
    assert len(history) > 0
    # Check that steps executed and telemetry recorded
    for step in history:
        assert "agent_pos" in step
        assert "friction" in step
        assert "remaining_energy" in step
