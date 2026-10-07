"""
Unit Test Suite for Geometric Phase Memory & Topological Transducer Engine
(tests/test_geometric_phase_dynamics.py)
"""

import pytest
import math
import torch
from core.topology.geometric_phase_memory_engine import (
    GeometricPhaseMemoryEngine,
    AttractorWell,
    MemoryRetrievalResult,
    StateTransitionResult,
    SelfHealingResult,
    MetaFrameResetResult
)


def test_geometric_phase_memory_engine_initialization():
    engine = GeometricPhaseMemoryEngine(dimension=16, stress_threshold=10.0)
    assert engine.dimension == 16
    assert engine.threshold == 10.0
    assert engine.horizon_ratio == 0.5
    assert len(engine.attractors) == 0


def test_register_attractor_and_holographic_retrieval():
    engine = GeometricPhaseMemoryEngine(dimension=16, stress_threshold=10.0)

    pattern_a = torch.randn(16)
    pattern_b = torch.randn(16)

    engine.register_attractor("attractor_alpha", pattern_a)
    engine.register_attractor("attractor_beta", pattern_b)

    assert len(engine.attractors) == 2

    # Query with a partial noisy sample of pattern_a
    noisy_query = pattern_a + torch.randn(16) * 0.1
    ret_res = engine.retrieve_memory(noisy_query)

    assert isinstance(ret_res, MemoryRetrievalResult)
    assert ret_res.attractor_name == "attractor_alpha"
    assert ret_res.resonance_score > 0.5
    assert ret_res.is_phase_locked is True
    assert ret_res.retrieved_rotor.shape == (5, 5)


def test_1tan_stress_calculation_and_gauge_smoothing():
    engine = GeometricPhaseMemoryEngine(dimension=16, epsilon_gauge=1e-5, stress_threshold=10.0)
    phase_vec = torch.randn(16) * 0.5

    stress_vec, stress_mag = engine.calculate_1tan_stress(phase_vec)

    assert stress_vec.shape == (16,)
    assert not torch.isnan(stress_vec).any()
    assert not torch.isinf(stress_vec).any()
    assert stress_mag > 0.0


def test_state_transition_via_phase_slip():
    engine = GeometricPhaseMemoryEngine(dimension=16, stress_threshold=100.0)

    pattern_a = torch.ones(16)
    pattern_b = -torch.ones(16)

    engine.register_attractor("attractor_alpha", pattern_a)
    engine.register_attractor("attractor_beta", pattern_b)

    # Initial lock on alpha
    engine.retrieve_memory(pattern_a)

    # Apply shock below threshold
    mild_shock = torch.zeros(16)
    trans_mild = engine.step_state_transition(mild_shock)
    assert trans_mild.status == "STRESS_ACCUMULATING"
    assert trans_mild.phase_slip_occurred is False

    # Apply large shock exceeding stress threshold -> Phase-Slip transition
    engine.threshold = 1.0
    strong_shock = torch.ones(16) * (math.pi / 2 - 0.1)
    trans_strong = engine.step_state_transition(strong_shock)
    assert trans_strong.status == "PHASE_SLIP_TRANSITION"
    assert trans_strong.phase_slip_occurred is True
    assert trans_strong.transition_energy > 0.0


def test_3stage_autonomous_self_healing():
    engine = GeometricPhaseMemoryEngine(dimension=16, stress_threshold=10.0)

    pattern_a = torch.randn(16)
    engine.register_attractor("attractor_alpha", pattern_a)
    engine.retrieve_memory(pattern_a)

    # Introduce disturbance impulse noise
    noise_impulse = torch.randn(16) * 0.5
    heal_res = engine.self_heal_noise(noise_impulse)

    assert isinstance(heal_res, SelfHealingResult)
    assert heal_res.initial_disturbance_norm > 0.0
    assert heal_res.restoring_force_norm > 0.0
    assert heal_res.dynamic_damping > engine.gamma_0
    assert heal_res.healed_rotor.shape == (5, 5)


def test_synesthetic_frequency_fusion():
    engine = GeometricPhaseMemoryEngine(dimension=16, stress_threshold=10.0)

    vision = torch.randn(8, 8)
    audio = torch.randn(16)
    tactile = torch.randn(16)
    science = torch.randn(16)

    fused_wave = engine.fuse_synesthetic_frequencies(vision, audio, tactile, science)

    assert fused_wave.shape == (16,)
    assert pytest.approx(float(torch.norm(fused_wave).item()), 0.001) == 1.0


def test_recursive_meta_frame_reset():
    engine = GeometricPhaseMemoryEngine(dimension=16, stress_threshold=2.0)

    # Normal stress -> No reset
    reset_normal = engine.trigger_recursive_meta_frame_reset(accumulated_stress=1.0)
    assert reset_normal.is_meta_triggered is False
    assert reset_normal.new_horizon_ratio == 0.5

    # Overload stress > threshold * 1.5 -> Trigger meta frame reset
    reset_overload = engine.trigger_recursive_meta_frame_reset(accumulated_stress=4.0)
    assert reset_overload.is_meta_triggered is True
    assert reset_overload.new_horizon_ratio > 0.5
    assert not torch.equal(reset_overload.reconfigured_anchor, torch.zeros(3))
