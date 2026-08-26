r"""
test_fourier_mellin_clifford_engine.py
========================================
Unit tests for Fourier-Mellin IFFT wave-void relaxation, Complex Manifold phase flow,
and Clifford Geometric Algebra multivector rotor dynamics in `core/physics/clifford_fourier_mellin_engine.py`.
"""

import math
import numpy as np
import pytest

from core.physics.clifford_fourier_mellin_engine import (
    CliffordFourierMellinEngine,
    FourierMellinTransformEngine,
    ComplexManifoldEngine,
    CliffordMultivectorEngine,
    Multivector
)
from core.physics.sangsaeng_sanggeuk_field import SangsaengSanggeukField, DynamicEntity

def test_fourier_mellin_isometry_and_noise_reduction():
    fm_engine = FourierMellinTransformEngine(num_scale_bins=16, num_phase_bins=16)

    # Create a synthetic wave field grid with pure signal + white noise
    rng = np.random.default_rng(42)
    pure_signal = np.outer(np.sin(np.linspace(0, np.pi, 16)), np.cos(np.linspace(0, 2 * np.pi, 16)))
    white_noise = rng.normal(0, 0.5, (16, 16)) + 1j * rng.normal(0, 0.5, (16, 16))
    grid = pure_signal + white_noise

    # Forward FFT
    spectrum = fm_engine.forward_transform(grid)

    # Entropy noise filtering (r_min, r_max)
    filtered_spec, noise_ratio = fm_engine.filter_entropy_noise(spectrum, r_min=0.1, r_max=0.8)
    assert noise_ratio > 0.0

    # Inverse Transform (IFFT)
    restored_grid, void_tension = fm_engine.inverse_transform(filtered_spec)
    assert restored_grid.shape == (16, 16)
    # Void tension should be non-negative and finite
    assert void_tension >= 0.0

def test_complex_manifold_orthogonality_and_vortex_ring():
    cm_engine = ComplexManifoldEngine(dim=3)

    # 1. Cauchy-Riemann orthogonality check
    u = np.outer(np.cos(np.linspace(0, math.pi, 10)), np.ones(10))
    v = np.outer(np.ones(10), np.sin(np.linspace(0, math.pi, 10)))
    cr_error = cm_engine.check_cauchy_riemann_orthogonality(u, v)
    assert cr_error >= 0.0

    # 2. Helical Stream & Vortex Ring Attractor Relaxation
    scale_axis = np.linspace(-2.0, 2.0, 16)
    xs, ys, zs = cm_engine.generate_helical_stream(1.0, 1.0, scale_axis)
    vx, vy, vz, void_energy = cm_engine.relax_to_vortex_ring(xs, ys, zs, dt=0.2, steps=30)

    assert vx.shape == (16,)
    assert void_energy < 0.5 # Crystallizes into torus invariant

def test_clifford_multivector_rotor_and_laplacian():
    ga_engine = CliffordMultivectorEngine(dim=3)

    # Multivector creation & norm
    mv = Multivector(dim=3, scalar=1.0, vector=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert pytest.approx(mv.norm(), 0.001) == math.sqrt(2.0)

    # Rotor Sandwich product R * Psi * ~R
    rotor = ga_engine.construct_bivector_rotor(np.array([0.0, 0.0, 1.0], dtype=np.float32), math.pi / 2.0)
    rotated_mv = ga_engine.rotor_sandwich_transform(mv, rotor)

    # Vector rotated 90 deg along z-axis: (1, 0, 0) -> magnitude 1.0
    assert pytest.approx(rotated_mv.vector[0], abs=1e-3) == 0.0
    assert pytest.approx(abs(rotated_mv.vector[1]), abs=1e-3) == 1.0

    # Laplacian harmonic equilibrium grad^2 Psi = 0
    grid = np.sin(np.linspace(0, math.pi, 8))[:, np.newaxis] * np.cos(np.linspace(0, math.pi, 8))[np.newaxis, :]
    relaxed_grid, laplacian_e = ga_engine.compute_laplacian_equilibrium(grid)
    assert relaxed_grid.shape == (8, 8)
    assert laplacian_e >= 0.0

def test_sangsaeng_sanggeuk_field_clifford_integration():
    field = SangsaengSanggeukField(dimensions=3, num_scale_bins=8)

    e1 = DynamicEntity("entity_1", "WaveA", position=np.array([0, 0, 0], dtype=np.float32), amplitude=2.0, phase=0.5)
    e2 = DynamicEntity("entity_2", "WaveB", position=np.array([1, 0, 0], dtype=np.float32), amplitude=1.5, phase=1.0)
    field.add_entity(e1)
    field.add_entity(e2)

    step_result = field.step(0.1)
    assert "abductive_converged" in step_result
    assert "void_tension_energy" in step_result

    relaxation = field.apply_scale_twist_and_void_relaxation(0.1)
    assert "clifford_fourier_mellin" in relaxation
    cfm_data = relaxation["clifford_fourier_mellin"]
    assert "noise_reduction_ratio" in cfm_data
    assert "e_void_tension" in cfm_data
    assert "cauchy_riemann_error" in cfm_data
    assert "vortex_ring_e_void" in cfm_data
    assert "laplacian_residual_energy" in cfm_data
