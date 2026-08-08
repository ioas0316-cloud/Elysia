import numpy as np
import pytest
from core.sensory.experiential_language_mapper import (
    ExperientialLanguageMapper,
    OpticalCausalDynamicsPipeline,
    PhysicalSensationProfile,
    HomeostasisDeficit,
    ExperienceType
)

def test_optical_destructive_interference_superposition():
    """
    Verify Wave Superposition & Destructive Phase Cancellation in the pipeline.
    """
    resolution = 100
    pipeline = OpticalCausalDynamicsPipeline(resolution=resolution)

    # Simple sine wave representations
    t = np.linspace(0, 1.0, resolution)
    ext_wave = np.sin(2 * np.pi * 5 * t).astype(np.float32)
    # Exact opposite phase to simulate cancellation
    int_wave = -np.sin(2 * np.pi * 5 * t).astype(np.float32)

    superposed = pipeline.destructive_interference(ext_wave, int_wave)

    # Perfect cancellation should yield a zero-amplitude wave
    assert np.allclose(superposed, 0.0, atol=1e-5)


def test_pure_prime_frequency_survival():
    """
    Verify that prime frequencies survive/resonate and non-prime frequencies
    are heavily attenuated based on their divisor counts (destructive phase cancellation).
    """
    resolution = 50
    pipeline = OpticalCausalDynamicsPipeline(resolution=resolution)

    # Let our input spectrum be a flat wave of ones across all frequencies
    flat_spectrum = np.ones(resolution, dtype=np.float32)

    extracted = pipeline.extract_prime_residuals(flat_spectrum, lambda_coef=1.0, gamma_resonance=0.5)

    # Verify that prime frequency bins (e.g. 2, 3, 5, 7, 11) survived and resonated (value > 1.0)
    for p in [2, 3, 5, 7, 11, 13, 17, 19]:
        if p < resolution:
            assert extracted[p] == pytest.approx(1.5)

    # Verify that non-primes are attenuated based on divisor count
    # 4 has divisors [1, 2, 4] -> divisor count = 3 -> exp(-1.0 * 3) = 0.0498
    assert extracted[4] == pytest.approx(float(np.exp(-3.0)), rel=1e-5)

    # 12 has divisors [1, 2, 3, 4, 6, 12] -> count = 6 -> exp(-1.0 * 6) = 0.00247
    assert extracted[12] == pytest.approx(float(np.exp(-6.0)), rel=1e-5)


def test_spatial_curvature_projection():
    """
    Verify that spatial curvature κ(x) correctly projects the double difference (second derivative),
    mapping prime residuals into distinct spatial curves (peaks and troughs).
    """
    resolution = 100
    pipeline = OpticalCausalDynamicsPipeline(resolution=resolution)

    # Form a wave with a single local peak (quadratic bump)
    # Psi = -x^2 -> Double derivative = -2.0
    x = np.linspace(-5.0, 5.0, resolution)
    residuals = -np.square(x).astype(np.float32)

    curvature = pipeline.project_spatial_curvature(residuals, alpha=1.0)

    # For a quadratic function, double difference should be a constant negative value
    # Except at the boundaries where second derivative boundary transitions are smoothed
    inner_curvature = curvature[5:-5]
    assert np.all(inner_curvature < 0.0)
    # The double difference for -x^2 with step dx is constant: res[i+1] - 2*res[i] + res[i-1]
    # Since res is quadratic, the second difference should be exactly constant
    assert np.allclose(inner_curvature, inner_curvature[0], rtol=1e-3)


def test_trajectory_alignment_bvp():
    """
    Verify that trajectory alignment under applied optical boundary conditions (BVP)
    warps the past and future trajectories toward the boundary geodesic (Reverse Causalization).
    """
    resolution = 100
    pipeline = OpticalCausalDynamicsPipeline(resolution=resolution)

    past_traj = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    future_traj = [(3.0, 4.0), (4.0, 5.0)]

    # Flat boundary condition (no intensity) -> no alignment warping should occur
    zero_boundary = np.zeros(resolution, dtype=np.float32)
    aligned_past, aligned_future = pipeline.align_trajectory_bvp(past_traj, future_traj, zero_boundary)

    assert aligned_past == past_traj
    assert aligned_future == future_traj

    # High-intensity boundary condition -> significant warping should occur
    intense_boundary = np.ones(resolution, dtype=np.float32) * 5.0
    warped_past, warped_future = pipeline.align_trajectory_bvp(past_traj, future_traj, intense_boundary, mu=0.5)

    assert warped_past != past_traj
    assert warped_future != future_traj

    # Verify warping behaves continuously and smoothly
    for i, (y_orig, x_orig) in enumerate(past_traj):
        y_warp, x_warp = warped_past[i]
        assert isinstance(y_warp, float)
        assert isinstance(x_warp, float)
        # Verify it has drifted toward the alignment attractor center (at least one axis changed)
        assert (y_warp != y_orig) or (x_warp != x_orig)


def test_mapper_optical_dynamics_end_to_end():
    """
    Verify the end-to-end integration of Optical Causal Dynamics in ExperientialLanguageMapper.
    """
    mapper = ExperientialLanguageMapper(resolution=100)

    # Initial state
    initial_love = mapper.homeostasis.love
    initial_order = mapper.homeostasis.order
    initial_resistor = mapper.variable_resistor.resistance

    # Input external wave representing raw sensory light/perturbation
    external_stimulus = np.sin(np.linspace(0, 2 * np.pi, 100)).astype(np.float32)

    # Process optical dynamics interference
    result = mapper.process_optical_interference(external_stimulus, lambda_coef=0.5, gamma_resonance=0.3)

    assert "superposed" in result
    assert "prime_residuals" in result
    assert "curvature" in result
    assert "aligned_past" in result
    assert "aligned_future" in result

    # Check continuous coupled impacts on homeostasis and physical state
    assert mapper.homeostasis.love > initial_love
    assert mapper.homeostasis.order < initial_order
    assert mapper.variable_resistor.resistance != initial_resistor

    # Verify that the standing wave memory has incorporated prime residual patterns smoothly
    assert np.any(mapper.standing_wave_memory > 0.0)
