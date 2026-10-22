"""
test_tri_system_adaptation.py

pytest suite for ELYSIA Multi-Scale Tri-System Comparative Adaptation Engine
(demos/layer_2/demo_tri_system_structural_adaptation.py)
"""

import math
import pytest
from demos.layer_2.demo_tri_system_structural_adaptation import (
    UnifiedMultiScaleWaveEnvironment,
    FixedCognitiveSystem,
    ParameterAdaptiveSystem,
    RelationalStructureAdaptiveSystem,
    run_tri_system_experiment
)


def test_fixed_cognitive_system_1_behavior():
    """Verify Fixed Cognitive System (System 1) behavior under low vs high mismatch."""
    env = UnifiedMultiScaleWaveEnvironment(num_micro=16, base_freq=1.0, seed=42)
    sys1 = FixedCognitiveSystem(input_dim=16, tolerance=0.3)

    # Low mismatch
    w_low = env.generate_wavefield(t=1.0, phase_mismatch=0.01, scale_shift=1.0, noise_level=0.01)
    res_low = sys1.process_wavefield(w_low)
    assert res_low["rejected"] is False
    assert res_low["mode"] == "Forced Fitting"
    assert res_low["action_possibility"] > 0.0

    # Extreme mismatch exceeding tolerance
    w_high = env.generate_wavefield(t=2.0, phase_mismatch=math.pi * 0.95, scale_shift=2.0, noise_level=0.05)
    res_high = sys1.process_wavefield(w_high)
    assert res_high["rejected"] is True
    assert res_high["mode"] == "Exclusion/Rejection"
    assert res_high["action_possibility"] == 0.0
    assert sys1.structural_modifications_count == 0


def test_parameter_adaptive_system_2_behavior():
    """Verify Parameter Adaptive System (System 2) updates parameters while keeping topology fixed."""
    env = UnifiedMultiScaleWaveEnvironment(num_micro=16, base_freq=1.0, seed=42)
    sys2 = ParameterAdaptiveSystem(input_dim=16, learning_rate=0.1, seed=101)

    initial_offsets = list(sys2.internal_phase_offsets)
    w = env.generate_wavefield(t=1.5, phase_mismatch=0.2, scale_shift=1.1, noise_level=0.02)
    res = sys2.process_wavefield(w)

    assert res["coupling_dim"] == 2
    assert sys2.structural_modifications_count == 0
    assert sys2.parameter_adjustments_count == 1
    # Check that phase offsets drifted away from initial zeros
    assert any(abs(a - b) > 1e-5 for a, b in zip(sys2.internal_phase_offsets, initial_offsets))


def test_relational_structure_system_3_expansion():
    """Verify Relational Structure Adaptive System (System 3) expands coupling dimension under structural conflict."""
    env = UnifiedMultiScaleWaveEnvironment(num_micro=16, base_freq=1.0, seed=42)
    sys3 = RelationalStructureAdaptiveSystem(
        input_dim=16, adaptation_threshold=0.20, pruning_threshold=0.15, noise_limit=0.35, seed=202
    )

    w_conflict = env.generate_wavefield(t=2.5, phase_mismatch=math.pi * 0.85, scale_shift=1.5, noise_level=0.02)
    res = sys3.process_wavefield(w_conflict)

    assert res["expanded_this_step"] is True
    assert res["coupling_dim"] == 3
    assert sys3.structural_modifications_count == 1
    assert len(sys3.growth_ring_history) == 1
    assert sys3.growth_ring_history[0]["type"] == "EXPANSION"


def test_relational_structure_system_3_noise_defense():
    """Verify System 3 resists false structural expansion under extreme stochastic noise boundary."""
    env = UnifiedMultiScaleWaveEnvironment(num_micro=16, base_freq=1.0, seed=42)
    sys3 = RelationalStructureAdaptiveSystem(
        input_dim=16, adaptation_threshold=0.20, pruning_threshold=0.15, noise_limit=0.35, seed=202
    )

    # Feed high noise wavefields
    for step in range(5):
        w_noise = env.generate_wavefield(t=5.0 + step * 0.1, phase_mismatch=0.0, scale_shift=1.0, noise_level=0.95)
        res = sys3.process_wavefield(w_noise)

    assert res["estimated_noise"] > sys3.noise_limit
    assert res["coupling_dim"] == 2  # Dimension should NOT expand due to noise
    assert sys3.structural_modifications_count == 0


def test_relational_structure_system_3_pruning():
    """Verify System 3 prunes redundant coupling dimensions when mismatch strain subsides."""
    env = UnifiedMultiScaleWaveEnvironment(num_micro=16, base_freq=1.0, seed=42)
    sys3 = RelationalStructureAdaptiveSystem(
        input_dim=16, adaptation_threshold=0.20, pruning_threshold=0.15, noise_limit=0.35, seed=202
    )

    # Step 1: Trigger expansion
    for step in range(3):
        w_shift = env.generate_wavefield(t=10.0 + step * 0.1, phase_mismatch=math.pi * 0.9, scale_shift=1.5, noise_level=0.01)
        _ = sys3.process_wavefield(w_shift)

    assert sys3.coupling_dim > 2

    # Step 2: Environmental recovery
    for step in range(10):
        w_rec = env.generate_wavefield(t=12.0 + step * 0.1, phase_mismatch=0.0, scale_shift=1.0, noise_level=0.0)
        res_rec = sys3.process_wavefield(w_rec)

    assert sys3.pruning_count >= 1
    assert sys3.coupling_dim == 2
    assert sys3.growth_ring_history[-1]["type"] == "PRUNING_CONSOLIDATION"


def test_deterministic_reproducibility():
    """Verify that identical seeds produce bitwise reproducible results across independent runs."""
    results_run1, summary_run1 = run_tri_system_experiment()
    results_run2, summary_run2 = run_tri_system_experiment()

    assert summary_run1["sys1_modifications"] == summary_run2["sys1_modifications"]
    assert summary_run1["sys2_parameter_adjustments"] == summary_run2["sys2_parameter_adjustments"]
    assert summary_run1["sys3_structural_expansions"] == summary_run2["sys3_structural_expansions"]
    assert summary_run1["sys3_pruning_count"] == summary_run2["sys3_pruning_count"]
    assert len(summary_run1["sys3_growth_rings"]) == len(summary_run2["sys3_growth_rings"])

    for r1, r2 in zip(results_run1, results_run2):
        assert math.isclose(r1["sys3"]["relational_loss"], r2["sys3"]["relational_loss"], rel_tol=1e-9)
