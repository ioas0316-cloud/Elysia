"""
Unit test suite for Topological Spectral Engine & Spectral Wave Relaxation Modules.
"""

import math
import torch
import pytest
from synaptic_architecture.topological_spectral_engine import (
    TopologicalSpectralEngine,
    SpectralWaveRelaxation1D,
    SpectralWaveRelaxation2D,
    SpectralWaveRelaxation3D,
    TopologicalAttractorRelaxation,
    SpectralAttractorRelaxation,
    StaticRotorField2D,
    quaternion_mul,
    quaternion_conjugate,
    create_rotor,
)


def test_topological_spectral_engine_encode_and_resonance():
    engine = TopologicalSpectralEngine(shape=(32, 32), device='cpu')
    assert engine.field.shape == (32, 32)
    assert engine.field.dtype == torch.complex64

    # Encode a wave mode with frequency (2, 3) and phase pi/4
    engine.encode_data_wave(frequency_k=(2, 3), phase_phi=math.pi / 4, amplitude=2.0)
    assert not torch.all(engine.field == 0)

    # Query by resonance with matching probe phase
    res_signal = engine.query_by_resonance(probe_phase=math.pi / 4)
    assert res_signal.shape == (32, 32)
    assert (res_signal >= 0).all()  # relu non-negative


def test_topological_spectral_engine_operator():
    engine = TopologicalSpectralEngine(shape=(16, 16), device='cpu')
    engine.encode_data_wave(frequency_k=(1, 1), phase_phi=0.0, amplitude=1.0)

    # Transfer function: simple low pass mask
    H = torch.ones((16, 16), dtype=torch.complex64)
    engine.apply_spectral_operator(H)
    assert engine.field.shape == (16, 16)


def test_spectral_wave_relaxation_modules():
    # 1D
    spec1d = SpectralWaveRelaxation1D(in_channels=4, length=64)
    x1d = torch.randn(2, 4, 64)
    out1d = spec1d(x1d)
    assert out1d.shape == (2, 4, 64)

    # 2D
    spec2d = SpectralWaveRelaxation2D(in_channels=8, height=32, width=32)
    x2d = torch.randn(2, 8, 32, 32)
    out2d = spec2d(x2d)
    assert out2d.shape == (2, 8, 32, 32)

    # 3D
    spec3d = SpectralWaveRelaxation3D(in_channels=4, depth=16, height=16, width=16)
    x3d = torch.randn(2, 4, 16, 16, 16)
    out3d = spec3d(x3d)
    assert out3d.shape == (2, 4, 16, 16, 16)


def test_attractor_relaxation():
    # Topological Attractor
    topo_attr = TopologicalAttractorRelaxation(embed_dim=16)
    x = torch.randn(2, 32, 16)
    out_topo = topo_attr(x, relax_steps=2)
    assert out_topo.shape == (2, 32, 16)

    # Spectral Attractor
    spec_attr = SpectralAttractorRelaxation(embed_dim=16)
    out_spec = spec_attr(x)
    assert out_spec.shape == (2, 32, 16)


def test_quaternion_and_static_rotor_field():
    # Rotor creation
    axis = torch.tensor([0.0, 1.0, 0.0])
    angle = torch.tensor(math.pi / 2)
    rotor = create_rotor(axis, angle)
    assert rotor.shape == (4,)

    rotor_conj = quaternion_conjugate(rotor)
    identity = quaternion_mul(rotor, rotor_conj)
    assert torch.allclose(identity, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-5)

    # Static Rotor Field 2D
    rotor_field = StaticRotorField2D(height=16, width=16)
    assert rotor_field.field.shape == (16, 16, 4)

    init_corner = rotor_field.field[0, 0].clone()
    init_center = rotor_field.field[8, 8].clone()

    # In-place local rotor update
    rotor_field.apply_local_rotor_update(center_h=8, center_w=8, radius=2, axis=axis, angle=angle)
    # Center should be updated away from initial state
    assert not torch.allclose(rotor_field.field[8, 8], init_center)
    # Corner should remain unchanged
    assert torch.allclose(rotor_field.field[0, 0], init_corner)

    # Wave diffusion
    rotor_field.relax_wave_diffusion(steps=3, dt=0.1)
    assert rotor_field.field.shape == (16, 16, 4)
