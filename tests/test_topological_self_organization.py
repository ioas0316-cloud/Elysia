r"""
Unit tests for TopologicalSelfOrganizationEngine, TopologicalDNA, and TemperamentProfile.
"""

import pytest
import torch
import numpy as np
from core.topology.fiber_bundle_manifold import FiberBundleManifold, SENSORY_PORTS
from core.topology.topological_self_organization import (
    TemperamentProfile,
    TopologicalDNA,
    TopologicalSelfOrganizationEngine
)


def test_temperament_profile_normalization_and_bias():
    profile = TemperamentProfile(gut=2.0, brain=1.0, heart=1.0)
    assert pytest.approx(profile.gut) == 0.5
    assert pytest.approx(profile.brain) == 0.25
    assert pytest.approx(profile.heart) == 0.25

    bias = profile.compute_sensory_port_bias()
    assert bias.shape == (5,)
    assert pytest.approx(float(bias.sum().item())) == 1.0


def test_enneagram_deformation_matrix():
    profile = TemperamentProfile(enneagram_type=8)  # Challenger: heavy frontal impact
    deform = profile.compute_enneagram_deformation_matrix(stress_level=0.5)
    assert deform.shape == (3, 3)
    assert deform[0, 0] > 1.0  # Axis 1 expanded for challenger type


def test_topological_dna_imprint():
    manifold = FiberBundleManifold(num_points=50)
    dna = TopologicalDNA(
        temperament=TemperamentProfile(gut=0.6, brain=0.2, heart=0.2, enneagram_type=1)
    )

    # Initial Euclidean metric norm before imprinting
    initial_metric = manifold.h_metric.clone()

    dna.imprint(manifold)

    # Check metric was deformed by primal attractors
    assert not torch.allclose(initial_metric, manifold.h_metric)
    # Check gauge potential was initialized
    assert float(torch.norm(manifold.gauge_A_t).item()) > 0.0


def test_sensory_wave_stream_and_temperament_filtering():
    gut_dna = TopologicalDNA(temperament=TemperamentProfile(gut=0.8, brain=0.1, heart=0.1))
    heart_dna = TopologicalDNA(temperament=TemperamentProfile(gut=0.1, brain=0.1, heart=0.8))

    engine_gut = TopologicalSelfOrganizationEngine(num_points=50, dna=gut_dna)
    engine_heart = TopologicalSelfOrganizationEngine(num_points=50, dna=heart_dna)

    wave = {"SOMATOSENSORY": 0.8, "VISION": 0.8}

    res_gut = engine_gut.receive_sensory_wave_stream(wave)
    res_heart = engine_heart.receive_sensory_wave_stream(wave)

    # Gut-heavy engine should amplify Somatosensory more than Heart-heavy engine
    assert res_gut["SOMATOSENSORY"] > res_heart["SOMATOSENSORY"]
    # Heart-heavy engine should amplify Vision more than Gut-heavy engine
    assert res_heart["VISION"] > res_gut["VISION"]


def test_hebbian_phase_plasticity_carving():
    engine = TopologicalSelfOrganizationEngine(num_points=100)
    wave = {"SOMATOSENSORY": 1.0, "VISION": 1.0}

    initial_deform_norm = float(torch.norm(engine.manifold.h_metric).item())

    # Stream wave and apply plasticity over multiple steps
    for _ in range(5):
        engine.receive_sensory_wave_stream(wave, dt=0.05)
        phase_matrix = engine.apply_hebbian_phase_plasticity(threshold=0.1, plasticity_rate=0.1)

    summary = engine.get_system_state_summary()
    assert summary["carved_valleys_count"] > 0
    assert summary["active_phase_lock_coherence"] > 0.0


def test_non_backprop_geodesic_deflection():
    engine = TopologicalSelfOrganizationEngine(num_points=50)

    initial_coords = engine.manifold.coords.clone()

    # Apply wave perturbation
    engine.receive_sensory_wave_stream({"SOMATOSENSORY": 1.0}, dt=0.02)

    # Step geodesic deflection without backward autograd
    accel = engine.step_non_backprop_geodesic_deflection(d_tau=0.02)

    assert accel.shape == (50, 4)
    # Spacetime coordinates should have moved along geodesic flow
    assert not torch.allclose(initial_coords, engine.manifold.coords)


def test_associative_domino_recall():
    engine = TopologicalSelfOrganizationEngine(num_points=50)
    wave = {"SOMATOSENSORY": 1.0, "VISION": 1.0}

    # Stream co-occurring wave to carve associative channel
    for _ in range(10):
        engine.receive_sensory_wave_stream(wave, dt=0.05)
        engine.apply_hebbian_phase_plasticity(threshold=0.1, plasticity_rate=0.1)

    # Stimulate ONLY SOMATOSENSORY port
    recall = engine.trigger_associative_domino_recall("SOMATOSENSORY", input_magnitude=1.0)

    assert recall["SOMATOSENSORY"] > 0.0
    # Vision should be recall-excited via phase lock coupling and metric conductance
    assert recall["VISION"] > 0.0


def test_epigenetic_lifetime_experience_compression():
    engine = TopologicalSelfOrganizationEngine(num_points=50)
    wave = {"SOMATOSENSORY": 1.0, "GUSTATION": 1.0}

    # Experience lifetime perturbations
    for _ in range(10):
        engine.receive_sensory_wave_stream(wave, dt=0.05)
        engine.apply_hebbian_phase_plasticity(threshold=0.1, plasticity_rate=0.1)

    # Epigenetically compress lifetime experience into child TopologicalDNA
    child_dna = engine.dna.compress_lifetime_experience(engine.manifold, top_k=3)

    assert isinstance(child_dna, TopologicalDNA)
    # Child should have inherited/expanded primal attractors
    assert len(child_dna.primal_attractors) >= len(engine.dna.primal_attractors)

    # Instantiate new engine for second generation
    gen2_engine = TopologicalSelfOrganizationEngine(num_points=50, dna=child_dna)
    summary2 = gen2_engine.get_system_state_summary()
    assert summary2["num_points"] == 50
