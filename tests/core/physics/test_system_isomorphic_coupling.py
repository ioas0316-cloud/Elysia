"""
Tests for SystemTumblerIsomorphicCouplingEngine.
Verifies:
1. Isomorphic coupling of System A (1) and System B (1) into System C (2*).
2. Tumbler (Ottogi) restoring dynamics under external displacement shock.
3. Morse potential attraction-repulsion tension equilibrium.
4. Absence vector and residual stress calculation for self-healing.
5. Two-way Body-Mind feedback and semantic mass / scar tensor updates.
"""

import pytest
import numpy as np
from L1_causal_field.core.physics.system_tumbler_isomorphic_coupling import (
    IsomorphicSystem,
    SystemTumblerIsomorphicCouplingEngine
)


def test_isomorphic_system_coupling():
    engine = SystemTumblerIsomorphicCouplingEngine(dimension=5)

    rotors_a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    rotors_b = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

    sys_a = IsomorphicSystem("System_A", scale=1.0, dimension=5, rotor_phases=rotors_a, semantic_mass=1.0)
    sys_b = IsomorphicSystem("System_B", scale=1.0, dimension=5, rotor_phases=rotors_b, semantic_mass=1.0)

    sys_c, info = engine.couple_isomorphic_systems(sys_a, sys_b, coupled_id="System_C_2star")

    assert sys_c.system_id == "System_C_2star"
    assert sys_c.scale == 2.0
    assert sys_c.semantic_mass > 2.0  # Increased by friction energy
    assert np.linalg.norm(sys_c.scar_tensor) > 0.0
    assert info["friction_energy"] > 0.0


def test_tumbler_ottogi_restoration():
    engine = SystemTumblerIsomorphicCouplingEngine(dimension=5, stiffness_k=10.0, damping_gamma=6.0)

    sys_c = IsomorphicSystem("System_C", scale=2.0, dimension=5, semantic_mass=2.5)

    # Apply strong external shock torque
    history = engine.apply_tumbler_restoration(sys_c, external_shock_torque=5.0, dt=0.01, num_steps=500)

    peak_tilt = max(abs(h["tilt_angle"]) for h in history)
    final_tilt = history[-1]["tilt_angle"]

    # Verify restoring dynamics: final tilt angle should converge back near 0 (< peak tilt and near 0)
    assert abs(final_tilt) < peak_tilt
    assert abs(final_tilt) < 0.05


def test_morse_tension_force():
    engine = SystemTumblerIsomorphicCouplingEngine(morse_de=1.0, morse_a=1.0, morse_r0=1.0)

    # Near distance (r < r0 = 1.0): repulsive force (> 0)
    f_near = engine.morse_tension_force(0.5)
    assert f_near > 0.0

    # Equilibrium distance: force near 0
    f_eq = engine.morse_tension_force(1.0)
    assert abs(f_eq) < 1e-5

    # Far distance (r > r0 = 1.0): attractive force (< 0)
    f_far = engine.morse_tension_force(2.0)
    assert f_far < 0.0


def test_absence_vector_and_residual_stress():
    engine = SystemTumblerIsomorphicCouplingEngine(dimension=5)

    systems = [
        IsomorphicSystem(f"Node_{i}", scale=1.0, dimension=5, rotor_phases=np.full(5, 0.2 * i))
        for i in range(4)
    ]

    # Compute absence vector for defect at index 2
    res = engine.compute_absence_and_residual_stress(systems, defect_index=2)

    assert "absence_vector" in res
    assert "residual_stress_tensor" in res
    assert res["residual_stress_norm"] > 0.0
    assert res["absence_vector"].shape == (5,)


def test_two_way_body_mind_feedback():
    engine = SystemTumblerIsomorphicCouplingEngine(dimension=5)
    sys = IsomorphicSystem("Self_System", scale=1.0, dimension=5, semantic_mass=1.5)

    initial_mass = sys.semantic_mass
    shock = np.array([0.5, -0.3, 0.4, -0.2, 0.1])

    res = engine.two_way_body_mind_feedback(sys, hardware_friction_shock=shock, dt=0.05)

    assert sys.semantic_mass > initial_mass
    assert res["mismatch_norm"] > 0.0
    assert res["scar_tensor_norm"] > 0.0
