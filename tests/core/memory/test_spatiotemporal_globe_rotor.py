"""
tests/core/memory/test_spatiotemporal_globe_rotor.py
=====================================================
Unit tests for InvariantArchetypeBackbone and SpatiotemporalGlobeRotorEngine.
"""

import pytest
import numpy as np
from L1_causal_field.core.memory.spectral_causal_visual_engine import (
    InvariantArchetypeBackbone,
    SpatiotemporalGlobeRotorEngine,
    CausalVisualReconstructionPipeline,
    SpectralAxisTransform,
)


def test_invariant_archetype_backbone_evaluation_and_synthesis():
    backbone = InvariantArchetypeBackbone(
        archetype_name="TestArchetype",
        reference_dimension=(32, 64),
        harmonic_frequencies=[1.0, 2.0, 4.0]
    )

    assert backbone.reference_manifold.shape == (32, 64)
    assert np.min(backbone.reference_manifold) >= 0.0
    assert np.max(backbone.reference_manifold) <= 1.0

    # Evaluate structural deviation against a random incoming field
    incoming_field = np.random.rand(32, 64)
    eval_res = backbone.evaluate_structural_deviation(incoming_field)

    assert eval_res["archetype_name"] == "TestArchetype"
    assert "deviation_magnitude" in eval_res
    assert "structural_alignment" in eval_res
    assert eval_res["archetype_friction_heat"] >= 0.0

    # Synthesize generative waveform without file loading
    synth_wave = backbone.synthesize_generative_waveform(
        rotation_angle=np.pi / 4,
        output_shape=(48, 96)
    )
    assert synth_wave.shape == (48, 96)
    assert np.min(synth_wave) >= 0.0
    assert np.max(synth_wave) <= 1.0


def test_spatiotemporal_globe_rotor_high_res_mapping():
    rotor_engine = SpatiotemporalGlobeRotorEngine(
        base_angular_velocity=1.5,
        vorticity_strength=2.0
    )

    # Simulated 4K resolution field (or high-res grid)
    high_res_field = np.random.rand(128, 128, 3)

    mapping_res = rotor_engine.map_high_res_to_spatiotemporal_manifold(
        high_res_field,
        target_globe_res=(64, 128)
    )

    assert mapping_res["globe_manifold"].shape == (64, 128)
    assert mapping_res["q_heat_total"] >= 0.0
    assert mapping_res["new_angular_velocity"] >= 1.0
    assert mapping_res["backbone_evaluation"]["structural_alignment"] <= 1.0

    # Test spatiotemporal rotation step
    step_res = rotor_engine.step_spatiotemporal_rotation(dt=0.2)
    assert step_res["angular_position"] > 0.0
    assert step_res["synthesized_manifold"].ndim == 2
    assert step_res["friction_heat_q"] >= 0.0


def test_causal_reconstruction_pipeline_with_globe_rotor():
    pipeline = CausalVisualReconstructionPipeline()
    gt_image = np.random.rand(64, 64, 3)

    res = pipeline.run_causal_reconstruction_pipeline(
        high_res_ground_truth=gt_image,
        scale_factor=2,
        noise_level=0.02
    )

    assert "reconstructed_field" in res
    assert res["metrics"]["psnr"] > 0.0
    assert res["metrics"]["ssim"] <= 1.0
