"""
Unit and Integration Tests for Neuro-Phase Causal Engine & Sensory Phase Coupling.

Tests:
1. Bidirectional Phase Negotiation & Error (q_err) Reduction.
2. Wave Entrainment across Text, Audio, and Vision modalities.
3. Phase State Transitions: GAS -> LIQUID -> ICE (Solid Crystal).
4. Storage and Retrieval of Crystallized Concept Attractor Basins.
"""

import pytest
import numpy as np
import math
from core.consciousness.neuro_phase_causal_engine import (
    NeuroPhaseCausalEngine,
    ExternalWaveStream,
    NeuroPhaseState
)
from core.sensory.multimodal_cognitive_frontend import MultimodalCognitiveFrontend


def test_bidirectional_phase_negotiation_q_err_reduction():
    engine = NeuroPhaseCausalEngine(num_nodes=16, lattice_dims=(4, 2, 2))
    engine.set_temperature(2.0)

    # Text stream represented as phase impulse wave
    text_phases = np.full(16, math.pi / 3.0)
    text_stream = ExternalWaveStream(
        modality="text",
        wave_phases=text_phases,
        frequencies=np.full(16, 40.0)
    )

    initial_result = engine.negotiate_bidirectional_phase(text_stream, coupling_gain=1.5)
    initial_q_err = initial_result.q_err

    # Perform multiple negotiation iterations to drive entrainment
    for _ in range(15):
        result = engine.negotiate_bidirectional_phase(text_stream, coupling_gain=2.0)

    final_q_err = result.q_err

    assert final_q_err < initial_q_err
    assert result.resonance_level > 0.8
    assert engine.system_temperature < 1.0  # Cools down during alignment


def test_multimodal_wave_entrainment_and_crystallization():
    engine = NeuroPhaseCausalEngine(num_nodes=16, lattice_dims=(4, 2, 2))

    # 1. Vision modality: 2D matrix
    vision_matrix = np.linspace(0, math.tau, 16).reshape((4, 4))
    vision_stream = ExternalWaveStream(
        modality="vision",
        wave_phases=vision_matrix,
        frequencies=np.full(16, 60.0)
    )

    for _ in range(25):
        res = engine.negotiate_bidirectional_phase(vision_stream, coupling_gain=2.5, crystallization_threshold=0.1)

    assert res.is_crystallized
    assert engine.global_phase_state == NeuroPhaseState.SOLID
    assert "vision_concept" in engine.crystallized_attractors


def test_phase_state_transition_sequence():
    engine = NeuroPhaseCausalEngine(num_nodes=16, lattice_dims=(4, 2, 2))

    # Initial state high temp
    engine.set_temperature(3.0)
    assert engine.global_phase_state == NeuroPhaseState.GAS

    # Mid temp
    engine.set_temperature(1.0)
    assert engine.global_phase_state == NeuroPhaseState.LIQUID

    # Phase Lock
    for node in engine.nodes.values():
        node.phase = math.pi / 2.0
    engine.last_q_err = 0.01
    engine.set_temperature(0.1)

    assert engine.global_phase_state == NeuroPhaseState.SOLID


def test_multimodal_cognitive_frontend_integration():
    frontend = MultimodalCognitiveFrontend(feature_dim=16)
    rgb_img = np.ones((10, 10, 3), dtype=np.uint8) * 200
    audio_wave = np.sin(np.linspace(0, 100, 100))
    text_input = "사과"

    res = frontend.process_multimodal_input(
        rgb_image=rgb_img,
        audio_wave=audio_wave,
        text_input=text_input
    )

    assert "axis_a_topology" in res
    assert "axis_b_qualia" in res
    assert abs(res["orthogonality_dot_product"]) < 1e-5
    assert res["phase_lock_order_parameter"] >= 0.0
