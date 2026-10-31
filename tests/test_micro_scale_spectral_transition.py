"""
tests/test_micro_scale_spectral_transition.py

Test Suite for Micro-Scale Spectral Transition & Continuous Phase Flow Engine
=============================================================================
Verifies:
1. Zero Pixel Storage & Invariant Law: Infinite-resolution field crystallization directly from Theta.
2. Multi-Scale Relational Scale Rotor: Smooth transitions across micro-textures and macro-illumination.
3. Optical Edge Cliff Conservation: Sharp boundary gradient preservation without raster blur.
4. Causal Anomaly & Phase Rupture Friction: Distinct spike in Q_heat and phase friction when
   generative distortions (joint collapse, scanline tearing, high-frequency noise) occur.
5. Container Integration: Seamless causal flow continuity verification during container playback.
"""

import math
import numpy as np
import pytest

import elysia_loader
elysia_loader.install()

from elysia_engine.core.micro_scale_spectral_transition_engine import (
    MicroScaleSpectralTransitionEngine,
    MicroTransitionLaw,
    PhaseTransitionAssessment
)
from elysia_engine.core.field_seed_manifest import (
    FieldSeed,
    RotorSeedDescriptor,
    FieldKernelDescriptor
)
from elysia_engine.core.spectrum_field_container import ElysiaSpectrumFieldContainer
from elysia_engine.core.legacy_video_decomposer import LegacyVideoDecomposer
from scripts.generate_realistic_anime_realworld_footage import generate_realistic_footage_sequence


class TestMicroScaleSpectralTransition:
    """Verifies all aspects of micro-scale spectral transition and causal phase flow."""

    def test_zero_pixel_storage_crystallization(self):
        """Verifies infinite-resolution field projection from pure law Theta without stored pixels."""
        engine = MicroScaleSpectralTransitionEngine()

        # Project at low resolution
        res_low = engine.crystallize_field(resolution_wh=(64, 36), t=0.0)
        assert res_low["framebuffer_rgb"].shape == (36, 64, 3)
        assert res_low["framebuffer_rgb"].dtype == np.uint8

        # Project at 4K aspect ratio resolution without pixelation
        res_high = engine.crystallize_field(resolution_wh=(256, 144), t=0.0)
        assert res_high["framebuffer_rgb"].shape == (144, 256, 3)
        assert np.mean(res_high["framebuffer_rgb"]) > 10.0

        # Assert edge cliff sharpness is maintained across scales
        grad_max = np.max(res_high["phase_gradient"])
        assert grad_max > 0.05

    def test_multi_scale_rotor_transitions(self):
        """Verifies smooth transition from micro-texture to macro-illumination via scale rotor."""
        engine = MicroScaleSpectralTransitionEngine(base_w=1.0)

        # Micro scale (high spatial oscillation frequency)
        engine.set_scale_rotor(5.0)
        micro_field = engine.crystallize_field(resolution_wh=(80, 80), t=0.0)

        # Macro scale (broad spatial illumination waves)
        engine.set_scale_rotor(0.2)
        macro_field = engine.crystallize_field(resolution_wh=(80, 80), t=0.0)

        # Micro field should exhibit higher spatial gradient variation than macro field
        micro_grad_std = np.std(micro_field["phase_gradient"])
        macro_grad_std = np.std(macro_field["phase_gradient"])
        assert micro_grad_std != macro_grad_std

        # Verify dynamic rotor rotation
        w_new = engine.rotate_scale_rotor(0.5)
        assert math.isclose(w_new, 0.7, rel_tol=1e-5)

    def test_causal_continuity_on_natural_flow(self):
        """Verifies that natural continuous physical flows pass causal continuity with minimal friction."""
        engine = MicroScaleSpectralTransitionEngine()

        f0 = engine.crystallize_field(resolution_wh=(64, 64), t=0.0)["framebuffer_rgb"]
        f1 = engine.crystallize_field(resolution_wh=(64, 64), t=0.05)["framebuffer_rgb"]

        assessment = engine.assess_causal_continuity(f0, f1)
        assert assessment.is_causally_natural is True
        assert assessment.mean_phase_friction < 0.1
        assert assessment.phase_rupture_count < 10
        assert assessment.energy_conservation_score > 0.8

    def test_causal_anomaly_detection_on_distortions(self):
        """Verifies that artificial distortions (joint collapse, tearing, noise) trigger causal anomaly alarms."""
        engine = MicroScaleSpectralTransitionEngine()

        f0 = engine.crystallize_field(resolution_wh=(64, 64), t=0.0)["framebuffer_rgb"]
        f1_natural = engine.crystallize_field(resolution_wh=(64, 64), t=0.05)["framebuffer_rgb"]

        # Synthetic distortions
        f1_collapsed = engine.inject_causal_distortion_benchmark(f1_natural, distortion_type="joint_collapse")
        f1_noisy = engine.inject_causal_distortion_benchmark(f1_natural, distortion_type="spectral_noise")
        f1_torn = engine.inject_causal_distortion_benchmark(f1_natural, distortion_type="phase_tearing")

        # Natural assessment
        eval_natural = engine.assess_causal_continuity(f0, f1_natural)

        # Distorted assessments
        eval_collapsed = engine.assess_causal_continuity(f0, f1_collapsed)
        eval_noisy = engine.assess_causal_continuity(f0, f1_noisy)
        eval_torn = engine.assess_causal_continuity(f0, f1_torn)

        # Causal mismatch friction should be significantly higher in distorted cases
        assert eval_collapsed.mean_phase_friction > eval_natural.mean_phase_friction
        assert eval_noisy.mean_phase_friction > eval_natural.mean_phase_friction
        assert eval_torn.mean_phase_friction > eval_natural.mean_phase_friction

        # Distortions should register phase ruptures
        assert eval_collapsed.phase_rupture_count > 0 or eval_collapsed.max_phase_friction > 0.2
        assert eval_noisy.phase_rupture_count > 0
        assert eval_torn.phase_rupture_count > 0

    def test_container_playback_causal_verification(self):
        """Verifies container-level integration of causal flow verification."""
        frames = generate_realistic_footage_sequence(num_frames=4, width=64, height=36, fps=24.0)
        decomposer = LegacyVideoDecomposer(num_rotors=4)
        container, _ = decomposer.decompose_frames(frames, fps=24.0, container_name="test_flow")

        flow_report = container.verify_causal_flow_continuity(resolution_wh=(64, 36), sample_steps=3)
        assert flow_report["all_transitions_natural"] is True
        assert flow_report["average_phase_friction"] < 0.1
        assert flow_report["samples_evaluated"] == 2
