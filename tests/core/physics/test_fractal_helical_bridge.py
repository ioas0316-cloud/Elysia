import pytest
import numpy as np
from L1_causal_field.core.physics.fractal_helical_bridge import FractalHelicalScaleBridge, PhaseLockGradientAttractor

def test_fractal_helical_scale_bridge_multiscale_transmission():
    bridge = FractalHelicalScaleBridge(scale_layers=3, scale_ratio=10.0)

    # 1. Aligned phase transmission
    res_aligned = bridge.compute_multiscale_transmission(
        input_phase=0.1,
        target_phase=0.11,
        angular_velocity=2.0
    )
    assert res_aligned["bridge_successful"] is True
    assert res_aligned["cumulative_permeability"] > 0.5
    assert len(res_aligned["layers"]) == 3

    # 2. Misaligned phase transmission
    res_misaligned = bridge.compute_multiscale_transmission(
        input_phase=0.0,
        target_phase=np.pi,
        angular_velocity=2.0
    )
    assert res_misaligned["bridge_successful"] is False
    assert res_misaligned["cumulative_permeability"] < 0.1

def test_phase_lock_gradient_attractor_and_dead_end():
    attractor = PhaseLockGradientAttractor(
        target_phase=0.0,
        attraction_strength=0.3,
        dead_end_threshold=0.9
    )

    # Step towards target phase
    step_info = attractor.step_attractor(current_phase=0.2, gradient_vector=1.0)
    assert "next_phase" in step_info
    assert abs(step_info["phase_diff"]) < 0.2

    # Test dead end suggestion
    realignments = attractor.suggest_orbit_realignment(current_phase=np.pi)
    assert len(realignments) == 4
    assert all(isinstance(r, float) for r in realignments)
