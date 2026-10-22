"""
tests/test_embodied_teleological_horizon_runtime.py

Comprehensive Unit and Integration Test Suite for:
1. Telos Purpose Kernel (0-Point Monad Attractor & Friction-to-Growth Transmutation)
2. Horizon Expansion Relational Spectrometer (Dynamic Axis Construction & Scale-Free Horizon Expansion N -> N+1)
3. Embodied Teleological Runtime (Body-Mind Scale Continuity and Integrated Perceptual Cycle)
"""

import pytest
import numpy as np

from elysia_core.teleological_purpose_kernel import TelosPurposeKernel
from elysia_core.horizon_expansion_spectrometer import HorizonExpansionSpectrometer
from elysia_core.embodied_teleological_runtime import EmbodiedTeleologicalRuntime


def test_telos_purpose_kernel_gravity_and_transmutation():
    """
    Tests Telos 0-point gravity attraction force and friction-to-growth transmutation engine.
    """
    kernel = TelosPurposeKernel(dimension=3, attractor_strength=0.5, growth_transmutation_rate=0.25)

    # Initial state away from origin
    state = np.array([0.1, 0.8, -0.5])
    gravity_force, dist = kernel.compute_zero_point_gravity(state)

    assert dist > 0.0
    assert len(gravity_force) == 3

    # Transmute cognitive friction
    torque = np.array([0.3, -0.4, 0.1])
    record = kernel.transmute_friction_to_growth(cognitive_friction=0.8, phase_torque=torque)

    assert record["gained_growth_energy"] == pytest.approx(0.2, abs=1e-4)
    assert kernel.accumulated_growth_energy > 0.0
    assert kernel.total_friction_processed == 0.8

    # Verify enhanced gravity force after growth energy accumulation
    enhanced_gravity, _ = kernel.compute_zero_point_gravity(state)
    assert np.linalg.norm(enhanced_gravity) > np.linalg.norm(gravity_force)


def test_horizon_expansion_spectrometer_bitter_taste_expansion():
    """
    Tests that introducing a completely novel taste ('bitter' / 쓴맛) into a 3-axis spectrum
    (sweet, sour, salty) causes phase differential tension exceeding threshold, triggering
    dynamic N -> N+1 axis construction (3D -> 4D).
    """
    spec = HorizonExpansionSpectrometer(
        initial_dimension=3,
        expansion_threshold=0.35,
        initial_axis_labels=["sweet", "sour", "salty"]
    )

    # Step 1: Input known 'sweet' vector -> low differential tension, no expansion
    sweet_vec = np.array([1.0, 0.0, 0.0])
    res_sweet = spec.analyze_and_expand_spectrum(sweet_vec, stimulus_label_hint="sweet")

    assert res_sweet["dimension"] == 3
    assert res_sweet["horizon_expanded"] is False
    assert res_sweet["matched_axis_label"] == "sweet"

    # Step 2: Input novel 'bitter' vector orthogonal to sweet, sour, salty (0.1, 0.1, 0.1, 0.95)
    # in current 3D space, this projects as a strong residual vector
    bitter_vec = np.array([0.05, 0.05, 0.05, 0.98])
    res_bitter = spec.analyze_and_expand_spectrum(bitter_vec, stimulus_label_hint="bitter")

    # Differential tension should exceed threshold and trigger expansion 3 -> 4
    assert res_bitter["horizon_expanded"] is True
    assert res_bitter["dimension"] == 4
    assert res_bitter["expansion_info"]["created_axis_label"] == "bitter"
    assert "bitter" in spec.axis_labels
    assert "bitter" in spec.facet_registry

    # Step 3: Re-input 'bitter' vector -> now recognized on newly constructed axis (low residual)
    res_bitter_2 = spec.analyze_and_expand_spectrum(bitter_vec, stimulus_label_hint="bitter")
    assert res_bitter_2["horizon_expanded"] is False
    assert res_bitter_2["dimension"] == 4
    assert res_bitter_2["matched_axis_label"] == "bitter"


def test_embodied_teleological_runtime_end_to_end_evolution():
    """
    Integration test verifying end-to-end perceptual processing, friction transmutation,
    and dynamic horizon expansion in EmbodiedTeleologicalRuntime.
    """
    runtime = EmbodiedTeleologicalRuntime(
        initial_dimension=3,
        expansion_threshold=0.35,
        friction_threshold=0.25,
        initial_axis_labels=["sweet", "sour", "salty"]
    )

    # 1. Perceive Light Sweet Stimulus
    step1 = runtime.perceive_and_evolve(
        stimulus_vector=np.array([1.0, 0.0, 0.0]),
        stimulus_label_hint="sweet",
        mechanism="light"
    )
    assert step1["dimension"] == 3
    assert step1["horizon_expanded"] is False
    assert step1["growth_energy"] > 0.0

    # 2. Perceive Crayon Sour Stimulus (Initial encounter with crayon -> high friction -> cellular bifurcation)
    step2 = runtime.perceive_and_evolve(
        stimulus_vector=np.array([0.0, 1.0, 0.0]),
        stimulus_label_hint="sour",
        mechanism="crayon"
    )
    assert step2["bifurcation_occurred"] is True
    assert step2["growth_energy"] > step1["growth_energy"]

    # 3. Perceive Novel 'Bitter' Waveform (Causes high residual tension -> triggers horizon expansion 3D -> 4D)
    bitter_stimulus = np.array([0.1, 0.1, 0.1, 1.2])
    step3 = runtime.perceive_and_evolve(
        stimulus_vector=bitter_stimulus,
        stimulus_label_hint="bitter",
        mechanism="pastel"
    )

    assert step3["horizon_expanded"] is True
    assert step3["dimension"] == 4
    assert step3["created_axis_label"] == "bitter"
    assert "bitter" in step3["active_axis_labels"]
    assert len(step3["telos_combined_vector"]) == 4

    # 4. Re-perceive 'Bitter' -> Seamless alignment with new axis
    step4 = runtime.perceive_and_evolve(
        stimulus_vector=bitter_stimulus,
        stimulus_label_hint="bitter",
        mechanism="pastel"
    )
    assert step4["horizon_expanded"] is False
    assert step4["dimension"] == 4
    assert step4["matched_axis_label"] == "bitter"
