"""
tests/test_optical_blister_rotor.py

Unit and Integration Tests for Optical Spectrum Bivector Rotor & Shadow Blister Effect Engine.
"""

import math
import numpy as np
import pytest
import sys

from elysia_engine.core.optical_blister_rotor import (
    OpticalSpectrumState,
    OpticalSpectrumBivectorRotor,
    ShadowBlisterTensionKernel,
    MicroReflexBlisterNetwork
)
from elysia_engine.core.hierarchical_blister_bridge import (
    HierarchicalBlisterBridge,
    BridgePropagationReport
)


def test_optical_spectrum_bivector_rotor_initialization():
    """Tests optical wavelength mapping to Triad weights and bivector properties."""
    # Red dominant (~650nm)
    rotor_red = OpticalSpectrumBivectorRotor(wavelength_nm=650.0)
    assert rotor_red.wavelength_nm == 650.0
    assert rotor_red.triad_weights[0] > rotor_red.triad_weights[1]  # Red > Green
    assert rotor_red.triad_weights[0] > rotor_red.triad_weights[2]  # Red > Blue

    # Blue dominant (~450nm)
    rotor_blue = OpticalSpectrumBivectorRotor(wavelength_nm=450.0)
    assert rotor_blue.triad_weights[2] > rotor_blue.triad_weights[0]  # Blue > Red

    # Bivector magnitude and angular frequency
    bv = rotor_red.get_bivector()
    assert len(bv) == 3
    assert np.linalg.norm(bv) > 0.0
    assert rotor_red.penumbra_radius > 0.0


def test_rotor_step_quaternion_evolution():
    """Tests Spin(3) rotor evolution under bivector angular velocity."""
    rotor = OpticalSpectrumBivectorRotor(wavelength_nm=550.0)
    q_init = rotor.rotor.copy()
    initial_phase = rotor.phase_theta

    rotor.step(dt=0.1)
    assert rotor.phase_theta != initial_phase
    assert not np.allclose(rotor.rotor, q_init)
    assert np.isclose(np.linalg.norm(rotor.rotor), 1.0, atol=1e-5)


def test_shadow_blister_non_linear_snap_attraction():
    """
    Tests that tension exhibits a sharp non-linear snap jump once penumbra overlap
    crosses the critical threshold Omega_c.
    """
    kernel = ShadowBlisterTensionKernel(omega_c=0.3, kappa_0=1.0, kappa_1=10.0, gamma_power=2.0)

    # Far apart (no overlap)
    r1 = OpticalSpectrumBivectorRotor(wavelength_nm=600.0, position=[0.0, 0.0, 0.0])
    r2 = OpticalSpectrumBivectorRotor(wavelength_nm=600.0, position=[5.0, 0.0, 0.0])
    res_far = kernel.evaluate_blister_interaction(r1, r2)
    assert res_far["overlap"] == 0.0
    assert res_far["force_total"] == 0.0
    assert not res_far["is_blister_active"]

    # Mild overlap below threshold (e.g., overlap ~ 0.2)
    # r1.penumbra_radius ~ 1.2, r2 ~ 1.2 -> r_sum = 2.4. dist = 2.0 -> overlap = (2.4 - 2.0)/2.4 = 0.166
    r2.position = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    res_sub = kernel.evaluate_blister_interaction(r1, r2)
    assert not res_sub["is_blister_active"]
    assert res_sub["force_total"] > 0.0

    # Overlap crossing threshold Omega_c = 0.3 (dist = 1.2 -> overlap = (2.4 - 1.2)/2.4 = 0.5)
    r2.position = np.array([1.2, 0.0, 0.0], dtype=np.float32)
    res_snap = kernel.evaluate_blister_interaction(r1, r2)
    assert res_snap["is_blister_active"]
    # Force should exhibit sharp non-linear increase due to kappa_1 term
    assert res_snap["force_total"] > res_sub["force_total"] * 3.0


def test_phase_interference_attraction_vs_repulsion():
    """Tests constructive (in-phase) vs destructive (out-of-phase) blister interaction."""
    kernel = ShadowBlisterTensionKernel(omega_c=0.2)
    r1 = OpticalSpectrumBivectorRotor(wavelength_nm=600.0, position=[0.0, 0.0, 0.0])
    r2 = OpticalSpectrumBivectorRotor(wavelength_nm=600.0, position=[1.0, 0.0, 0.0])

    # Constructive (Delta phi = 0)
    r1.phase_theta = 0.0
    r2.phase_theta = 0.0
    res_const = kernel.evaluate_blister_interaction(r1, r2)
    assert res_const["interference"] > 0.9
    assert res_const["force_total"] > 0.0  # Attractive

    # Destructive (Delta phi = pi)
    r2.phase_theta = math.pi
    res_dest = kernel.evaluate_blister_interaction(r1, r2)
    assert res_dest["interference"] < -0.9
    assert res_dest["force_total"] < 0.0  # Repulsive / dispersive


def test_micro_reflex_network_clumping():
    """Tests micro-scale reflex network clumping and cluster synthesis."""
    net = MicroReflexBlisterNetwork()
    # Add 3 resonant rotors in close proximity
    r1 = OpticalSpectrumBivectorRotor(wavelength_nm=500.0, position=[0.0, 0.0, 0.0])
    r2 = OpticalSpectrumBivectorRotor(wavelength_nm=505.0, position=[0.3, 0.0, 0.0])
    r3 = OpticalSpectrumBivectorRotor(wavelength_nm=510.0, position=[0.0, 0.3, 0.0])
    net.add_rotor(r1)
    net.add_rotor(r2)
    net.add_rotor(r3)

    # Run reflex simulation
    for _ in range(30):
        rep = net.step(dt=0.03)

    # Rotors should be drawn closer
    d12 = np.linalg.norm(r1.position - r2.position)
    assert d12 < 0.35

    # Synthesize macro-cluster rotor
    macro = net.synthesize_cluster_rotor([0, 1, 2])
    assert 495.0 <= macro.wavelength_nm <= 515.0
    assert macro.penumbra_scale > 1.0


def test_hierarchical_blister_bridge_gating_and_macro_coupling():
    """
    Tests that HierarchicalBlisterBridge successfully gates information
    and propagates coherent states to Helmholtz and 3x3x3 fractal nodes.
    """
    bridge = HierarchicalBlisterBridge(r_gate=0.6, grid_shape=(16, 16, 16))

    r1 = OpticalSpectrumBivectorRotor(wavelength_nm=650.0, position=[0.0, 0.0, 0.0])
    r2 = OpticalSpectrumBivectorRotor(wavelength_nm=652.0, position=[0.2, 0.0, 0.0])
    bridge.register_rotor(r1)
    bridge.register_rotor(r2)

    # Initial synchronized phase -> micro_coherence is high, passing R_gate
    rep = bridge.step(dt=0.02)
    assert rep.is_gated is True
    assert rep.macro_wave_modes is not None
    assert "scalar_trace" in rep.macro_wave_modes
    assert "vector_shear" in rep.macro_wave_modes
    assert "tensor_tt" in rep.macro_wave_modes
    # Sum of mode energy ratios should equal ~ 1.0
    total_energy = sum(rep.macro_wave_modes.values())
    assert np.isclose(total_energy, 1.0, atol=1e-4)

    # 3x3x3 fractal space coherence is updated
    assert 0.0 <= rep.fractal_coherence <= 1.0
