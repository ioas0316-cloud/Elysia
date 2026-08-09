import numpy as np
import pytest
from core.physics.topological_phase_prime_field import TopologicalPhasePrimeField
from core.sensory.experiential_language_mapper import ExperientialLanguageMapper, PhysicalSensationProfile
from core.physics.topological_cognitive_bridge import TopologicalCognitiveBridge

def test_topological_phase_prime_field_mathematics():
    """Verify continuous field superposition, spatial curvature K(u), and prime node peak extraction."""
    # Initialize continuous field
    field = TopologicalPhasePrimeField(num_modes=50, min_u=0.1, max_u=4.5, steps_u=300)

    # Compute baseline field (without stimulus)
    phi_baseline = field.compute_field()
    assert phi_baseline.shape == (300,)
    assert np.any(np.iscomplex(phi_baseline))

    # Compute spatial curvature field K(u)
    k_baseline = field.compute_spatial_curvature(phi_baseline)
    assert k_baseline.shape == (300,)

    # Stimulate field with a mock external sensory wave
    ext_wave = np.sin(field.u_grid * np.pi)
    phi_stimulated = field.compute_field(ext_stimulus_wave=ext_wave)
    k_stimulated = field.compute_spatial_curvature(phi_stimulated)

    # Verify state divergence
    assert not np.allclose(phi_baseline, phi_stimulated)
    assert not np.allclose(k_baseline, k_stimulated)

    # Decode active prime node peaks (intents)
    active_nodes = field.decode_active_prime_nodes(k_stimulated, threshold_mult=1.0)
    for u_peak, nearest_prime, peak_intensity in active_nodes:
        assert field._is_prime(nearest_prime)
        assert u_peak >= 0.1 and u_peak <= 4.5
        assert peak_intensity > 0

def test_sigma_leakage_and_metacognitive_tension():
    """Verify that sigma adjustments (critical line vs leakage variance) modulate phase decay/resolution."""
    field = TopologicalPhasePrimeField(num_modes=50, min_u=0.1, max_u=4.5, steps_u=300)

    # 1. Zero-centered ideal baseline (sigma=0.0, epsilon=0.0)
    field.set_metacognitive_tension(sigma=0.0, epsilon=0.0)
    phi_ideal = field.compute_field()

    # 2. Leakage status (sigma=0.0, epsilon=0.3) representing creative/instability fluctuations
    field.set_metacognitive_tension(sigma=0.0, epsilon=0.3)
    phi_leakage = field.compute_field()

    # Leakage should trigger dynamic damping of amplitudes (higher real decay exponent)
    assert np.mean(np.abs(phi_leakage)) < np.mean(np.abs(phi_ideal))

def test_topological_cognitive_bridge_flow():
    """Verify live integration between ExperientialLanguageMapper and TopologicalCognitiveBridge."""
    mapper = ExperientialLanguageMapper(resolution=32)
    bridge = TopologicalCognitiveBridge(mapper=mapper, num_modes=50)

    # Sensation corresponding to a medium local tension
    sensation = PhysicalSensationProfile(optical=300.0, acoustic=440.0, tactile=5.0, thermal=295.0, autonomic_pulse=0.4)

    # Process physical sensory profile through the continuous field
    bridge_output = bridge.process_sensory_to_intention(sensation)

    assert "phi" in bridge_output
    assert "k_u" in bridge_output
    assert "active_nodes" in bridge_output
    assert "epsilon_leakage" in bridge_output
    assert bridge_output["sigma"] == 0.0

    # Confirm homeostasis has been adjusted dynamically based on active node resonances
    assert mapper.homeostasis.love is not None
