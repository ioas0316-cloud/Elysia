"""
Unit & Integration Tests for Integrated Scale Cognition Loop,
Continuous Wavelength Spectral Transform, Toroidal Vortex Rotor,
and Conceptual Unit System Integration.
"""

import pytest
import numpy as np
import torch

from L1_causal_field.core.memory.spectral_causal_visual_engine import SpectralAxisTransform
from L1_causal_field.core.sovereignty.toroidal_vortex_rotor import ToroidalVortexRotorEngine
from L1_causal_field.core.consciousness.conceptual_unit_system_engine import ConceptualUnitSystemEngine
from L1_causal_field.core.consciousness.integrated_scale_cognition_loop import IntegratedScaleCognitionLoop


class TestContinuousSpectralAndCognitionPipeline:
    """Test suite verifying scale-invariant continuous sensory-cognitive coupling."""

    def test_spectral_axis_transform_q_heat_and_chromatic(self):
        """Verify continuous spectral projection, Q_heat = |∇Φ|^1.5, and chromatic triad self-classification."""
        transformer = SpectralAxisTransform()

        # Create a 2D continuous visual field Φ(x, y)
        x = np.linspace(-1.0, 1.0, 32)
        y = np.linspace(-1.0, 1.0, 32)
        xx, yy = np.meshgrid(x, y)
        field_2d = np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)

        spectral_field = transformer.project_to_spectral_axis(field_2d)
        continuity = transformer.classify_spectral_continuity(spectral_field)

        assert "q_heat_map" in continuity
        assert "total_friction_heat" in continuity
        assert "chromatic_triad" in continuity

        q_heat_map = continuity["q_heat_map"]
        grads = np.gradient(spectral_field)
        expected_grad_mag = np.sqrt(sum(g**2 for g in grads))
        expected_q_heat = expected_grad_mag ** 1.5

        np.testing.assert_allclose(q_heat_map, expected_q_heat, rtol=1e-5)
        assert len(continuity["chromatic_triad"]) == 3
        assert np.isclose(np.sum(continuity["chromatic_triad"]), 1.0) or np.linalg.norm(continuity["chromatic_triad"]) > 0

    def test_toroidal_vortex_rotor_heat_recycling(self):
        """Verify ToroidalVortexRotorEngine recycles Q_heat and chromatic gain into rotational velocity."""
        rotor = ToroidalVortexRotorEngine(base_rotational_velocity=1.0, vorticity_strength=1.0)
        initial_vel = rotor.rotational_velocity

        chromatic_red_flux = np.array([0.8, 0.1, 0.1])
        res = rotor.recycle_friction_and_mismatch(
            q_heat=2.5,
            sensory_mismatch_delta=0.2,
            chromatic_triad=chromatic_red_flux
        )

        assert res["new_rotational_velocity"] > initial_vel
        assert res["chromatic_gain"] > 1.0
        assert rotor.accumulated_recycled_heat > 0.0

    def test_conceptual_unit_system_q_heat_coupling(self):
        """Verify ConceptualUnitSystemEngine maps Q_heat into tactile tension and produces empathy resonance."""
        engine = ConceptualUnitSystemEngine(dimension=32)
        result = engine.process_conceptual_unit(
            raw_input="Continuous Scale-Invariant Truth",
            q_heat=5.0,
            input_chromatic=np.array([0.6, 0.3, 0.1])
        )

        assert result.sensory_profile.tactile_tension > 0.1
        assert result.auditory_empathy_resonance >= 0.0
        assert "R_flux" in result.chromatic_triad
        assert "B_order" in result.chromatic_triad
        assert "Y_entropy" in result.chromatic_triad

    def test_integrated_scale_cognition_loop_cycle(self):
        """Verify end-to-end IntegratedScaleCognitionLoop cycle with multimodal inputs and continuous field coupling."""
        loop = IntegratedScaleCognitionLoop(
            feature_dim=16,
            dim_micro=32,
            dim_meso=64,
            dim_macro=128
        )

        rgb_image = np.random.uniform(0, 255, size=(16, 16, 3))
        audio_wave = np.sin(np.linspace(0, 440 * 2 * np.pi, 200))
        text_input = "Scale Boundary Layer Continuity"

        res = loop.perceive_and_integrate(
            rgb_image=rgb_image,
            audio_wave=audio_wave,
            text_input=text_input
        )

        assert res.friction_heat_q >= 0.0
        assert res.vortex_rotor_velocity > 1.0
        assert res.conceptual_unit_result is not None
        assert res.overall_resonance >= 0.0
        assert res.unified_waveform_field is not None
