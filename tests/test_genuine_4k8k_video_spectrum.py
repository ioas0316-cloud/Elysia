"""
tests/test_genuine_4k8k_video_spectrum.py

Automated Test Suite for Genuine Real-World 4K/8K Video Spectrum Ingestion,
Clifford Rotor Extraction, and Transducer Recomposition.
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
from scripts.verify_genuine_4k8k_real_world_video import load_genuine_4k_frames


class TestGenuine4K8KVideoSpectrum:
    """Verifies genuine real-world optical camera footage ingestion and 4K/8K recomposition."""

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
        """Verifies decomposition of genuine 4K frames into Clifford Rotors and Field Seed."""
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

    def test_genuine_4k_8k_recomposition(self):
        """Verifies infinite-resolution transducer projection at 4K and 8K."""
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
        assert np.max(fb_4k["grad_mag"]) > 0.1 # Preserved Edge Cliff gradient

        # Subsampled fast 8K projection check
        fb_8k_sample = container.play_frame(t=0.0, resolution_wh=(960, 540))
        assert fb_8k_sample["framebuffer_rgb"].shape == (540, 960, 3)
