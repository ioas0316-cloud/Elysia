"""
Unit and Integration Tests for Dressmaker Phase Lock Runtime & Layer 3 Knowledge Bus Slot
========================================================================================
"""

import numpy as np
import pytest
from core.knowledge_base.knowledge_bus_slot import KnowledgeBusSlot, KnowledgeChunk
from elysia_core.dressmaker_phase_lock_runtime import (
    DressmakerPhaseLockRuntime,
    FabricWaveSpectrum,
    CharacterBodyRotor,
)


def test_knowledge_bus_slot_operations():
    """Tests slotting in/out chunks and aggregate field calculation."""
    bus = KnowledgeBusSlot(slot_id="TEST_SLOT", vector_dim=5)
    assert len(bus.connected_chunks) == 0
    assert np.all(bus.aggregate_knowledge_field == 0)

    chunk1 = KnowledgeChunk(
        chunk_id="C1",
        title="Test Chunk 1",
        source_type="test",
        content_text="Sample content 1",
        semantic_vector=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        constraint_order=1.0
    )
    bus.slot_in(chunk1)
    assert len(bus.connected_chunks) == 1
    assert np.allclose(bus.aggregate_knowledge_field, [1.0, 2.0, 3.0, 4.0, 5.0])

    chunk2 = KnowledgeChunk(
        chunk_id="C2",
        title="Test Chunk 2",
        source_type="test",
        content_text="Sample content 2",
        semantic_vector=np.array([3.0, 0.0, 1.0, 0.0, 1.0]),
        constraint_order=1.0
    )
    bus.slot_in(chunk2)
    assert len(bus.connected_chunks) == 2
    # Average of [1,2,3,4,5] and [3,0,1,0,1] => [2, 1, 2, 2, 3]
    assert np.allclose(bus.aggregate_knowledge_field, [2.0, 1.0, 2.0, 2.0, 3.0])

    spectrum = bus.generate_phase_constraint_spectrum(sample_points=64)
    assert len(spectrum) == 64
    assert np.max(np.abs(spectrum)) <= 1.0

    unslotted = bus.slot_out("C1")
    assert unslotted.chunk_id == "C1"
    assert len(bus.connected_chunks) == 1


def test_fabric_wave_spectrum_factories():
    """Tests fabric spectrum factory methods."""
    silk = FabricWaveSpectrum.create_silk()
    assert silk.fabric_id == "FABRIC_SILK_01"
    assert silk.weave_density == 45.0
    assert silk.tension_tensor.shape == (3, 3)
    assert len(silk.fold_frequency_spectrum) == 64

    leather = FabricWaveSpectrum.create_armored_leather()
    assert leather.fabric_id == "FABRIC_LEATHER_01"
    assert leather.weave_density == 120.0
    assert leather.seam_resistance > silk.seam_resistance


def test_dressmaker_phase_locking():
    """Tests phase-locking runtime between fabric and character body rotor."""
    bus = KnowledgeBusSlot()
    chunk = KnowledgeChunk(
        chunk_id="C_NBOOK",
        title="Weave Constraint",
        source_type="notebook_lm",
        content_text="Content",
        semantic_vector=np.array([0.5, 0.5, -0.5, 0.2, 0.1]),
        constraint_order=1.0
    )
    bus.slot_in(chunk)

    runtime = DressmakerPhaseLockRuntime(knowledge_bus=bus)
    silk = FabricWaveSpectrum.create_silk("FABRIC_SILK_TEST")

    t_samples = np.linspace(0, 4 * np.pi, 64)
    body = CharacterBodyRotor(
        character_id="TEST_BODY_01",
        world_origin="ActionWorld",
        body_mass=70.0,
        rotor_angles=np.array([0.0, 0.0, 0.0]),
        body_surface_curvature=np.sin(t_samples) * 0.8
    )

    result = runtime.phase_lock_fabric_to_body(silk, body)

    assert result["fabric_id"] == "FABRIC_SILK_TEST"
    assert result["character_id"] == "TEST_BODY_01"
    assert 0.0 <= result["phase_coherence_score"] <= 1.0
    assert result["status"] in ["PHASE_LOCKED", "PHASE_DIVERGENT"]
    assert len(result["phase_locked_waveform"]) == 64

    # Test motion trajectory simulation
    key = result["coupling_key"]
    trajectory = runtime.simulate_movement_resonance(coupling_key=key, motion_impulse=1.5, time_steps=8)
    assert len(trajectory) == 8
    assert all(isinstance(x, float) for x in trajectory)
