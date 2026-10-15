"""
Unit and Integration Tests for Elysia 4-Stage Interoperability Conversion Bridge
"""

import numpy as np
import pytest

from elysia_core.interoperability_bridge import (
    DiscretePhasorEncoder,
    RotorFieldCoreMiddleware,
    InSituPhaseErrorFilter,
    WaveCollapseDecoder,
    InteroperabilityBridgePipeline
)
from elysia_core.unified_system_architecture import PhasorField


class TestDiscretePhasorEncoder:
    def test_encode_bytes(self):
        data = b"Elysia Wavefield"
        field = DiscretePhasorEncoder.encode_bytes(data)

        assert isinstance(field, PhasorField)
        assert field.shape == (len(data),)
        assert field.field.dtype == np.complex128
        assert np.all(field.amplitudes >= 0.5) and np.all(field.amplitudes <= 1.0)

    def test_encode_text(self):
        text = "지능공학 아키텍처"
        field = DiscretePhasorEncoder.encode_text(text)

        assert field.shape == (len(text),)
        assert field.field.dtype == np.complex128

    def test_encode_image(self):
        image = np.array([[0, 128, 255], [64, 192, 32]], dtype=np.uint8)
        field = DiscretePhasorEncoder.encode_image(image)

        assert field.shape == (2, 3)
        assert field.field.dtype == np.complex128


class TestRotorFieldCoreMiddleware:
    def test_middleware_dynamics_and_chunking(self):
        middleware = RotorFieldCoreMiddleware(chunking_threshold=0.5)
        # Create a coherent input field
        coherent_field = PhasorField((8,), field=np.ones(8, dtype=np.complex128))

        processed_field, promoted_rotors = middleware.process_field_dynamics(
            coherent_field, rotation_angle=np.pi / 4
        )

        assert processed_field.shape == (8,)
        assert len(promoted_rotors) > 0


class TestInSituPhaseErrorFilter:
    def test_filter_noise_and_rectify(self):
        error_filter = InSituPhaseErrorFilter()
        clean_field = PhasorField((10,), field=np.ones(10, dtype=np.complex128))

        # Add artificial phase noise
        noise = np.array([0.1, -0.2, 0.3, 0.0, -0.1, 0.4, 0.0, 0.2, -0.3, 0.1])
        noisy_field = clean_field.rotate(noise)

        rectified_field, diagnostics = error_filter.filter_noise_and_rectify(
            noisy_field, clean_field
        )

        assert rectified_field.shape == clean_field.shape
        assert diagnostics["global_distortion_level"] >= 0.0
        assert diagnostics["post_coherence"] >= diagnostics["pre_coherence"]


class TestWaveCollapseDecoder:
    def test_decode_bytes(self):
        original = b"Hello, Phasor!"
        field = DiscretePhasorEncoder.encode_bytes(original)
        decoded = WaveCollapseDecoder.decode_bytes(field, original_length=len(original))

        assert decoded == original

    def test_decode_text(self):
        original = "Elysia System"
        field = DiscretePhasorEncoder.encode_text(original)
        decoded = WaveCollapseDecoder.decode_text(field, original_length=len(original))

        assert decoded == original


class TestInteroperabilityBridgePipeline:
    def test_process_bytes_pipeline(self):
        pipeline = InteroperabilityBridgePipeline(chunking_threshold=0.5)
        raw_data = b"Seamless 4-Stage Bridge Test"

        decoded, metrics = pipeline.process_bytes_pipeline(
            raw_data, rotation_shift=np.pi / 6, noise_level=0.05
        )

        assert metrics["input_bytes_count"] == len(raw_data)
        assert metrics["output_bytes_count"] == len(raw_data)
        assert decoded == raw_data

    def test_process_image_pipeline(self):
        pipeline = InteroperabilityBridgePipeline(chunking_threshold=0.5)
        image = np.array([
            [10, 20, 30],
            [100, 150, 200],
            [50, 80, 240]
        ], dtype=np.uint8)

        decoded_image, metrics = pipeline.process_image_pipeline(
            image, rotation_shift=0.0, noise_level=0.02
        )

        assert decoded_image.shape == image.shape
        max_pixel_err = np.max(np.abs(decoded_image.astype(int) - image.astype(int)))
        assert max_pixel_err <= 5
