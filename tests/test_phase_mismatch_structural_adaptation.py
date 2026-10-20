#!/usr/bin/env python3
"""
tests/test_phase_mismatch_structural_adaptation.py

Unit and integration tests for ELYSIA Phase Mismatch & Structural Adaptation.

Verifies:
1. Mismatch Detection Reproducibility across multi-scale wave signals.
2. Structural Transformation Triggering when mismatch strain exceeds threshold.
3. Pre/Post Adaptation Behavioral Diffs (restoration of action possibility & loss reduction).
4. Generalization to Unseen Phase and Frequency Shifts.
5. Boundary and Noise Instability Conditions (over-adaptation checks).
6. Fair Comparison between Baseline System A and Adaptive System B under identical environments.
7. Growth Ring Graph Self-Tracing Consistency.
"""

import math
import pytest
from demo_phase_mismatch_structural_adaptation import (
    MultiScaleEnvironment,
    BaselineCognitiveSystemA,
    AdaptiveCognitiveSystemB,
    run_comparative_experiment,
)


def test_mismatch_detection_reproducibility():
    """Verifies that wavefield generation and phase mismatch strain detection are deterministic."""
    env = MultiScaleEnvironment(num_micro=16, base_freq=1.0, seed=42)
    wave1 = env.generate_wavefield(t=2.0, phase_mismatch=math.pi * 0.5, scale_shift=1.0)
    wave2 = env.generate_wavefield(t=2.0, phase_mismatch=math.pi * 0.5, scale_shift=1.0)

    # Determinism check
    for m1, m2 in zip(wave1[0], wave2[0]):
        assert math.isclose(m1, m2, abs_tol=1e-6)

    system_a = BaselineCognitiveSystemA(input_dim=16)
    res1 = system_a.process_wavefield(wave1)
    res2 = system_a.process_wavefield(wave2)

    assert math.isclose(res1["mismatch_strain"], res2["mismatch_strain"], abs_tol=1e-6)
    assert res1["mode"] == res2["mode"]


def test_baseline_system_a_behavior():
    """Verifies System A forces fitting for low mismatch and rejects high mismatch."""
    system_a = BaselineCognitiveSystemA(input_dim=16, tolerance=0.3)
    env = MultiScaleEnvironment(num_micro=16, base_freq=1.0, seed=100)

    # Low mismatch signal -> Forced Fitting
    wave_low = env.generate_wavefield(t=1.0, phase_mismatch=0.01, scale_shift=1.0, noise_level=0.01)
    res_low = system_a.process_wavefield(wave_low)
    assert res_low["rejected"] is False
    assert res_low["mode"] == "Forced Fitting"
    assert res_low["action_possibility"] > 0.0

    # Extreme orthogonal phase mismatch signal -> Rejection
    wave_high = env.generate_wavefield(t=2.5, phase_mismatch=math.pi * 0.6, scale_shift=1.2, noise_level=0.05)
    res_high = system_a.process_wavefield(wave_high)
    assert res_high["rejected"] is True
    assert res_high["mode"] == "Exclusion/Rejection"
    assert res_high["action_possibility"] == 0.0


def test_adaptive_system_b_structural_transformation():
    """Verifies System B expands coupling dimension and records growth ring when strain > threshold."""
    system_b = AdaptiveCognitiveSystemB(input_dim=16, adaptation_threshold=0.15, seed=101)
    initial_dim = system_b.coupling_dim
    assert initial_dim == 2

    env = MultiScaleEnvironment(num_micro=16, base_freq=1.0, seed=100)
    wave_mismatch = env.generate_wavefield(t=1.0, phase_mismatch=math.pi * 0.5, scale_shift=1.2, noise_level=0.05)

    res = system_b.process_wavefield(wave_mismatch)

    assert system_b.coupling_dim > initial_dim
    assert system_b.structural_modifications_count > 0
    assert len(system_b.growth_ring_history) == system_b.structural_modifications_count
    assert res["action_possibility"] > 0.5


def test_pre_post_adaptation_behavioral_diff():
    """Verifies that structural adaptation restores action possibility compared to fixed baseline."""
    env = MultiScaleEnvironment(num_micro=16, base_freq=1.0, seed=50)
    system_a = BaselineCognitiveSystemA(input_dim=16)
    system_b = AdaptiveCognitiveSystemB(input_dim=16, adaptation_threshold=0.2, seed=50)

    wave = env.generate_wavefield(t=3.0, phase_mismatch=math.pi * 0.7, scale_shift=1.3)

    res_a = system_a.process_wavefield(wave)
    res_b = system_b.process_wavefield(wave)

    # Under high phase mismatch, System B should significantly outperform System A in action possibility
    assert res_b["action_possibility"] > res_a["action_possibility"]
    assert res_b["relational_loss"] < res_a["relational_loss"] or res_a["rejected"]


def test_generalization_to_unseen_shift():
    """Verifies that once adapted, System B generalizes to unseen frequency/scale shifts."""
    env = MultiScaleEnvironment(num_micro=16, base_freq=1.0, seed=77)
    system_b = AdaptiveCognitiveSystemB(input_dim=16, adaptation_threshold=0.25, seed=77)

    # Initial wave triggering adaptation
    wave1 = env.generate_wavefield(t=1.0, phase_mismatch=math.pi * 0.5, scale_shift=1.2)
    _ = system_b.process_wavefield(wave1)

    # Unseen scale shift
    wave2 = env.generate_wavefield(t=4.0, phase_mismatch=math.pi * 0.5, scale_shift=2.5)
    res2 = system_b.process_wavefield(wave2)

    assert res2["action_possibility"] > 0.0
    assert res2["coupling_dim"] >= 3


def test_extreme_noise_boundary_conditions():
    """Verifies system behavior under extreme stochastic noise (instability check)."""
    env = MultiScaleEnvironment(num_micro=16, base_freq=1.0, seed=99)
    system_b = AdaptiveCognitiveSystemB(input_dim=16, adaptation_threshold=0.25, seed=99)

    extreme_noisy_wave = env.generate_wavefield(t=10.0, phase_mismatch=math.pi, noise_level=1.5)
    res = system_b.process_wavefield(extreme_noisy_wave)

    # Action possibility should degrade under extreme uncorrelative noise
    assert res["action_possibility"] < 0.8
    # Growth ring records the attempt to adapt
    assert len(system_b.growth_ring_history) > 0


def test_full_comparative_experiment():
    """Integration test running the full 4-scenario comparative suite."""
    system_a, system_b, results = run_comparative_experiment()

    assert "scenario_1" in results
    assert "scenario_2" in results
    assert "scenario_3" in results
    assert "scenario_4" in results

    assert system_b.structural_modifications_count > 0
    assert results["scenario_2"]["B"]["action_possibility"] > results["scenario_2"]["A"]["action_possibility"]
