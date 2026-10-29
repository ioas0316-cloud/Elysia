"""
tests/core/consciousness/test_spatiotemporal_integration.py
============================================================
Integration test for IntegratedScaleCognitionLoop spatiotemporal globe rotor dynamics.
"""

import pytest
import numpy as np
import torch
from L1_causal_field.core.consciousness.integrated_scale_cognition_loop import IntegratedScaleCognitionLoop


def test_integrated_scale_cognition_loop_spatiotemporal_rotor():
    loop = IntegratedScaleCognitionLoop()

    rgb_input = np.random.rand(64, 64, 3)
    audio_input = np.sin(np.linspace(0, 4 * np.pi, 256))

    cognition_res = loop.perceive_and_integrate(
        rgb_image=rgb_input,
        audio_wave=audio_input,
        text_input="Spatiotemporal Globe Rotor Integration Test",
    )

    assert cognition_res.spatiotemporal_globe_manifold is not None
    assert cognition_res.spatiotemporal_globe_manifold.shape == (128, 256)
    assert cognition_res.synthesized_generative_waveform is not None
    assert cognition_res.archetype_alignment >= 0.0
    assert cognition_res.friction_heat_q > 0.0
    assert cognition_res.vortex_rotor_velocity > 0.0

    diag = loop.get_diagnostics()
    assert diag["total_steps"] == 1
