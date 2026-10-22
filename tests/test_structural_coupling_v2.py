"""
tests/test_structural_coupling_v2.py

Comprehensive Unit and Integration Test Suite for Generative Mechanism Prismatic Differentiation,
Cognitive Friction, Phase Torque, and Cellular Lens Bifurcation in ELYSIA's Structural Coupling Runtime.
"""

import pytest
import numpy as np
from elysia_core.structural_coupling_runtime import (
    StructuralCouplingRuntime,
    GenerativeMechanismSynthesizer,
    MetaPlasticPrismaticLens,
    ResonantAttractorCoupling,
    CrossModalManifoldProjection
)


def test_generative_mechanism_synthesizer():
    """
    Tests that GenerativeMechanismSynthesizer synthesizes distinct physical constraint
    waveforms for identical surface yellow colors across Light, Crayon, and Pastel mechanisms.
    """
    surface_yellow = np.array([1.0, 1.0, 0.0])

    light_prof = GenerativeMechanismSynthesizer.synthesize_waveform("light", surface_yellow)
    crayon_prof = GenerativeMechanismSynthesizer.synthesize_waveform("crayon", surface_yellow)
    pastel_prof = GenerativeMechanismSynthesizer.synthesize_waveform("pastel", surface_yellow)

    # 1. Surface colors must be identical
    np.testing.assert_array_equal(light_prof["surface_color"], surface_yellow)
    np.testing.assert_array_equal(crayon_prof["surface_color"], surface_yellow)
    np.testing.assert_array_equal(pastel_prof["surface_color"], surface_yellow)

    # 2. Physical constraint profiles must differ decisively
    assert light_prof["viscosity"] < crayon_prof["viscosity"]
    assert pastel_prof["micro_roughness"] > light_prof["micro_roughness"]
    assert pastel_prof["phase_jitter"] > crayon_prof["phase_jitter"]

    # 3. Waveforms must have distinct energy profiles
    assert not np.array_equal(light_prof["waveform"], crayon_prof["waveform"])
    assert not np.array_equal(crayon_prof["waveform"], pastel_prof["waveform"])


def test_phase_torque_and_cognitive_friction_differentiation():
    """
    Tests that initial un-differentiated lens produces phase torque and cognitive friction
    when encountering differing generative waveforms for identical surface colors.
    """
    lens = MetaPlasticPrismaticLens(dimension=5, friction_threshold=0.25)
    obs_vec = np.array([0.8, 0.6, 0.4, 0.2, 0.1])

    light_prof = GenerativeMechanismSynthesizer.synthesize_waveform("light")
    crayon_prof = GenerativeMechanismSynthesizer.synthesize_waveform("crayon")
    pastel_prof = GenerativeMechanismSynthesizer.synthesize_waveform("pastel")

    tau_light, f_light = lens.compute_phase_torque_and_friction(light_prof, obs_vec)
    tau_crayon, f_crayon = lens.compute_phase_torque_and_friction(crayon_prof, obs_vec)
    tau_pastel, f_pastel = lens.compute_phase_torque_and_friction(pastel_prof, obs_vec)

    # Light emission has minimal friction; Crayon and Pastel have high cognitive friction
    assert f_light < f_crayon
    assert f_crayon > 0.5
    assert f_pastel > 0.5
    assert len(tau_crayon) == 5


def test_cellular_lens_bifurcation_and_friction_decay():
    """
    Tests that high cognitive friction triggers cellular lens bifurcation, and
    re-encountering the same generative mechanism produces seamless zero friction (learning completion).
    """
    runtime = StructuralCouplingRuntime(target_dim=5, friction_threshold=0.25)

    # Step 1: Perceive Light Yellow (minimal friction)
    step_light_1 = runtime.perceive_generative_stimulus("light")
    assert step_light_1["cognitive_friction"] < 0.1
    assert step_light_1["bifurcated"] is False

    # Step 2: Perceive Crayon Yellow (initial encounter -> high friction -> triggers bifurcation)
    step_crayon_1 = runtime.perceive_generative_stimulus("crayon")
    assert step_crayon_1["cognitive_friction"] > 0.5
    assert step_crayon_1["bifurcated"] is True
    assert "crayon" in runtime.prismatic_lens.bifurcated_cores

    # Step 3: Re-perceive Crayon Yellow (aligned invariant core -> cognitive friction collapses to 0)
    step_crayon_2 = runtime.perceive_generative_stimulus("crayon")
    assert step_crayon_2["cognitive_friction"] < 0.05
    assert step_crayon_2["bifurcated"] is False

    # Step 4: Perceive Pastel Yellow (initial encounter -> high friction -> triggers second bifurcation)
    step_pastel_1 = runtime.perceive_generative_stimulus("pastel")
    assert step_pastel_1["cognitive_friction"] > 0.5
    assert step_pastel_1["bifurcated"] is True
    assert "pastel" in runtime.prismatic_lens.bifurcated_cores

    # Step 5: Re-perceive Pastel Yellow (aligned invariant core -> friction collapses)
    step_pastel_2 = runtime.perceive_generative_stimulus("pastel")
    assert step_pastel_2["cognitive_friction"] < 0.05
    assert step_pastel_2["bifurcated"] is False

    # Total bifurcated invariant cores = 2 (crayon & pastel)
    assert len(runtime.prismatic_lens.bifurcated_cores) == 2


def test_runtime_cross_modal_and_attractor_coupling():
    """
    Tests that StructuralCouplingRuntime integrates cross-modal projection and resonant attractor coupling.
    """
    runtime = StructuralCouplingRuntime(target_dim=5)

    # Multi-step cycle across various mechanisms
    for _ in range(3):
        runtime.perceive_generative_stimulus("light")
        runtime.perceive_generative_stimulus("crayon")
        runtime.perceive_generative_stimulus("pastel")

    assert runtime.step_count == 9
    assert len(runtime.trajectory_buffer) > 0

    latest_episode = runtime.learning_episodes[-1]
    assert "modal_coherence" in latest_episode
    assert "active_basin_idx" in latest_episode
    assert "invariant_core_stability" in latest_episode
    assert latest_episode["invariant_core_stability"] > 0.0
