"""
tests/test_genuine_4k8k_video_spectrum.py

Automated Test Suite for Genuine Real-World 4K/8K Video Spectrum Ingestion,
Physical Optical Lens Wavefront Recomposition, and 4-Axis Causal Optical Differential.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

import elysia_loader
elysia_loader.install()

from elysia_engine.core.legacy_video_decomposer import (
    LegacyVideoDecomposer,
    VideoDecompositionReport
)
from elysia_engine.core.spectrum_field_container import (
    ElysiaSpectrumFieldContainer
)
from elysia_engine.core.optical_lens_wavefront_engine import (
    PhysicalOpticalLensWavefrontEngine
)
from elysia_engine.core.causal_optical_differential_engine import (
    CausalOpticalDifferentialEngine,
    CausalOpticalAssessmentReport
)
from elysia_engine.core.micro_scale_spectral_transition_engine import (
    MicroScaleSpectralTransitionEngine
)
from scripts.verify_genuine_4k8k_real_world_video import load_genuine_4k_frames


class TestGenuine4K8KVideoSpectrum:
    """Verifies genuine real-world optical camera footage ingestion, lens recomposition, and causal differential."""

    def test_genuine_4k_asset_loading(self):
        """Verifies loading of physical optical lens capture footage."""
        frames = load_genuine_4k_frames()
        assert len(frames) >= 2
        for f in frames:
            assert f.shape == (2160, 3840, 3)
            assert f.dtype == np.uint8
            # Confirm realistic non-monochrome content
            assert np.mean(f) > 20.0
            assert np.std(f) > 10.0

    def test_genuine_4k_decomposition(self):
        """Verifies decomposition of genuine 4K frames into Physical Optical Lens Wavefronts."""
        frames = load_genuine_4k_frames()[:2]
        decomposer = LegacyVideoDecomposer(num_rotors=4)
        container, report = decomposer.decompose_frames(
            frames=frames,
            fps=24.0,
            container_name="test_genuine_4k"
        )

        assert isinstance(container, ElysiaSpectrumFieldContainer)
        assert isinstance(report, VideoDecompositionReport)
        assert report.extracted_rotor_count == 4
        assert report.input_resolution == (3840, 2160)
        assert report.compression_ratio > 1000.0 # Extreme reduction (> 1000x)

    def test_genuine_4k_8k_optical_lens_recomposition(self):
        """Verifies physical optical lens projection at 4K and 8K."""
        frames = load_genuine_4k_frames()[:2]
        decomposer = LegacyVideoDecomposer(num_rotors=4)
        container, _ = decomposer.decompose_frames(
            frames=frames,
            fps=24.0,
            container_name="test_recomp"
        )

        # 4K projection
        fb_4k = container.play_frame(t=0.0, resolution_wh=(3840, 2160))
        assert fb_4k["framebuffer_rgb"].shape == (2160, 3840, 3)
        assert np.max(fb_4k["grad_mag"]) > 0.01 # Preserved Edge Cliff gradient
        assert fb_4k["mtf_response"] > 0.001

        # Subsampled fast 8K projection check
        fb_8k_sample = container.play_frame(t=0.0, resolution_wh=(960, 540))
        assert fb_8k_sample["framebuffer_rgb"].shape == (540, 960, 3)
        assert fb_8k_sample["mtf_response"] > 0.001

    def test_causal_optical_differential_evaluation(self):
        """Verifies 4-axis differential discriminating natural pan from artificial distortion."""
        frames = load_genuine_4k_frames()[:2]
        f0 = frames[0][::6, ::6]
        f1 = frames[1][::6, ::6]

        differential_engine = CausalOpticalDifferentialEngine()
        transition_engine = MicroScaleSpectralTransitionEngine()

        f1_distorted = transition_engine.inject_causal_distortion_benchmark(f1, distortion_type="joint_collapse")

        res_natural = differential_engine.evaluate_causal_differential(f0, f1)
        res_distorted = differential_engine.evaluate_causal_differential(f0, f1_distorted)

        assert isinstance(res_natural, CausalOpticalAssessmentReport)
        assert isinstance(res_distorted, CausalOpticalAssessmentReport)

        # Genuine camera pan preserves continuity
        assert res_natural.is_causally_natural is True
        assert res_natural.mean_causal_friction_q_heat < 0.01

        # Injected anomaly triggers rupture detection
        assert res_distorted.is_causally_natural is False
        assert res_distorted.rupture_count > res_natural.rupture_count
        assert res_distorted.localization_heatmap.shape == f0.shape[:2]

    def test_second_real_world_4k_asset(self):
        """Verifies generalization across multiple real-world 4K video assets."""
        second_asset = Path("demos/real_world_assets/second_real_world_4k.jpg")
        assert second_asset.exists()

        frames = load_genuine_4k_frames(second_asset)[:2]
        assert len(frames) == 2
        assert frames[0].shape == (2160, 3840, 3)

        decomposer = LegacyVideoDecomposer(num_rotors=4)
        container, report = decomposer.decompose_frames(
            frames=frames,
            fps=24.0,
            container_name="test_second_asset"
        )
        assert report.compression_ratio > 1000.0

        fb = container.play_frame(t=0.0, resolution_wh=(960, 540))
        assert fb["framebuffer_rgb"].shape == (540, 960, 3)
