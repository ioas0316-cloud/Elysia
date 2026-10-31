"""
tests/test_realistic_video_spectrum.py

Automated Test Suite for Step 4:
- Elysia Spectrum Field Container (.esf) Serialization & Deserialization
- Legacy Video Decomposer: Clifford 4D Rotors & Helmholtz Wave Spectrum Inversion
- Relational Spectrum Knob Engine: Coupled Phase Transitions (Solid/Liquid/Gas)
- Infinite Resolution 4K FrameBuffer Projection & Edge Cliff Gradient Preservation
- Standalone Playback Runtime without External Media Decoders
"""

import os
import math
import numpy as np
import pytest

import elysia_loader
elysia_loader.install()

from scripts.generate_realistic_anime_realworld_footage import (
    generate_realistic_anime_landscape_frame,
    generate_realistic_footage_sequence
)
from elysia_engine.core.spectrum_field_container import (
    ElysiaSpectrumFieldContainer,
    ContainerHeader,
    TrajectoryKeyframe
)
from elysia_engine.core.legacy_video_decomposer import (
    LegacyVideoDecomposer,
    VideoDecompositionReport
)
from elysia_engine.core.relational_spectrum_knob_engine import (
    RelationalSpectrumKnobEngine,
    MaterialityPhase
)
from elysia_engine.core.field_seed_manifest import (
    FieldSeed,
    RotorSeedDescriptor,
    FieldKernelDescriptor
)


class TestRealisticVideoSpectrum:
    """Verifies all aspects of the self-executable spectrum field container and video decomposition."""

    def test_realistic_frame_generator(self):
        """Checks synthetic generation of realistic scenery footage."""
        frame = generate_realistic_anime_landscape_frame(width=160, height=90, t=0.5, character_pos_x=0.5)
        assert frame.shape == (90, 160, 3)
        assert frame.dtype == np.uint8
        # Ensure varied rich color palette (not monochrome/black)
        assert np.mean(frame) > 20.0
        assert np.std(frame) > 10.0

    def test_legacy_video_decomposition(self):
        """Verifies decomposition of video frames into Clifford rotors and Field Seed."""
        frames = generate_realistic_footage_sequence(num_frames=6, width=120, height=68, fps=24.0)
        decomposer = LegacyVideoDecomposer(num_rotors=4)
        container, report = decomposer.decompose_frames(frames, fps=24.0, container_name="test_anime")

        assert isinstance(container, ElysiaSpectrumFieldContainer)
        assert isinstance(report, VideoDecompositionReport)
        assert report.extracted_rotor_count == 4
        assert report.input_frame_count == 6
        assert report.compression_ratio > 10.0 # Extreme reduction in storage footprint
        assert len(container.seed.rotors) == 4
        assert len(container.trajectories) > 0

    def test_esf_container_serialization_roundtrip(self, tmp_path):
        """Verifies packaging, saving, and unpacking of standalone .esf container file."""
        frames = generate_realistic_footage_sequence(num_frames=4, width=80, height=45, fps=24.0)
        decomposer = LegacyVideoDecomposer(num_rotors=4)
        container, _ = decomposer.decompose_frames(frames, fps=24.0, container_name="roundtrip_test")

        file_path = str(tmp_path / "test_container.esf")
        packed_len = container.save_to_file(file_path)
        assert packed_len > 0
        assert os.path.exists(file_path)

        # Unpack from file
        loaded = ElysiaSpectrumFieldContainer.load_from_file(file_path)
        assert loaded.header.container_name == "roundtrip_test"
        assert len(loaded.seed.rotors) == 4
        assert loaded.header.resolution_independent is True

    def test_infinite_resolution_projection_and_edge_cliff(self):
        """Verifies arbitrary high-resolution transducer sampling without pixelation."""
        frames = generate_realistic_footage_sequence(num_frames=4, width=80, height=45, fps=24.0)
        decomposer = LegacyVideoDecomposer(num_rotors=4)
        container, _ = decomposer.decompose_frames(frames, fps=24.0, container_name="res_test")

        # Project at low res
        fb_low = container.play_frame(t=0.0, resolution_wh=(48, 48))
        assert fb_low["framebuffer_rgb"].shape == (48, 48, 3)

        # Project at 4K aspect ratio high res (super-resolution sampling)
        fb_high = container.play_frame(t=0.0, resolution_wh=(192, 108))
        assert fb_high["framebuffer_rgb"].shape == (108, 192, 3)

        # Confirm sharp Edge Cliff gradient is preserved
        cliff_max = np.max(fb_high["grad_mag"])
        assert cliff_max > 0.5

    def test_relational_knob_coupled_phase_transition(self):
        """Verifies gear coupling of knobs and triadic state transition (Ice -> Water -> Gas)."""
        knob_engine = RelationalSpectrumKnobEngine()

        # Initial state should be Solid Ice / Past
        p0 = knob_engine._reorganize_field()
        assert p0.materiality_phase == MaterialityPhase.SOLID_ICE_PAST

        # Rotate tension knob into liquid range (approx pi rad)
        p_liquid = knob_engine.rotate_knob("phase_tension", math.pi * 0.8)
        assert p_liquid.materiality_phase == MaterialityPhase.LIQUID_WATER_PRESENT

        # Rotate further into gas vapor range (approx 1.6 pi rad)
        p_gas = knob_engine.rotate_knob("phase_tension", math.pi * 0.9)
        assert p_gas.materiality_phase == MaterialityPhase.GAS_VAPOR_FUTURE

        # Rotate scale knob and verify gear coupling ripples to other knobs
        initial_tension = knob_engine.tension_knob.value
        knob_engine.rotate_knob("scale_w", 0.5)
        # Tension knob should have shifted due to coupling matrix
        assert knob_engine.tension_knob.value != initial_tension

    def test_autonomous_playback_sequence(self):
        """Verifies continuous sequence rendering without external video decoders."""
        frames = generate_realistic_footage_sequence(num_frames=4, width=64, height=36, fps=24.0)
        decomposer = LegacyVideoDecomposer(num_rotors=4)
        container, _ = decomposer.decompose_frames(frames, fps=24.0, container_name="playback_test")

        rendered_seq = container.render_sequence(num_frames=5, resolution_wh=(64, 64))
        assert len(rendered_seq) == 5
        for f in rendered_seq:
            assert f.shape == (64, 64, 3)
            assert f.dtype == np.uint8
