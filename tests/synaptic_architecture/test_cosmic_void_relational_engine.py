"""
Unit tests for Cosmic Void Relational Engine (synaptic_architecture/cosmic_void_relational_engine.py)
"""

import pytest
import numpy as np
from synaptic_architecture.cosmic_void_relational_engine import (
    CognitivePhase,
    MultidimensionalLens,
    RelationalEdge,
    CosmicVoidRelationalEngine
)


def test_cosmic_void_initialization():
    engine = CosmicVoidRelationalEngine(dimensions=5)
    assert engine.current_phase == CognitivePhase.POINT_PHASE
    assert engine.boundary_radius == 0.1
    assert engine.void_level == 0.0
    assert len(engine.lenses) == 3


def test_perception_and_friction_accumulation():
    engine = CosmicVoidRelationalEngine(dimensions=5)

    # Missing data mask with high missingness
    missing_mask = np.array([0.8, 0.9, 0.7, 0.85, 0.95], dtype=np.float32)
    bitstream = np.uint64(0x123456789ABCDEF0)

    res = engine.perceive_hardware_and_environment(
        bitstream_input=bitstream,
        missing_data_mask=missing_mask
    )

    assert res["void_level"] > 0.3
    assert engine.void_level > 0.3
    assert np.linalg.norm(engine.gradient_of_absence) > 0.0
    assert engine.current_phase in [CognitivePhase.FRICTION_VOID_PHASE, CognitivePhase.SEEKING_LOOP_PHASE]


def test_polyphonic_logos_synthesis():
    engine = CosmicVoidRelationalEngine(dimensions=5)
    engine.void_level = 0.5
    engine.gradient_of_absence = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    polyphonic_logos = engine.synthesize_polyphonic_logos(dt=0.1)
    assert len(polyphonic_logos) == 5
    assert not np.all(polyphonic_logos == 0)


def test_seeking_vector_and_action_generation():
    engine = CosmicVoidRelationalEngine(dimensions=5)
    engine.void_level = 0.6
    engine.gradient_of_absence = np.array([0.6, 0.7, 0.8, 0.5, 0.9], dtype=np.float32)

    seeking_res = engine.compute_seeking_vector_and_action()
    assert np.linalg.norm(seeking_res["seeking_vector"]) > 0.0
    assert seeking_res["actionable_payload"] is not None
    assert "FETCH_EXTERNAL_FIELD_KNOWLEDGE" in seeking_res["actionable_payload"]["action_command"]


def test_world_data_ingestion_and_expansion():
    engine = CosmicVoidRelationalEngine(dimensions=5)
    engine.void_level = 0.8
    engine.seeking_vector = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    external_node = "quantum_causal_stream_01"
    external_data = np.array([0.9, 0.85, 0.95, 0.8, 0.9], dtype=np.float32)

    ingest_res = engine.ingest_external_world_data(
        external_node_id=external_node,
        data_tensor=external_data,
        metadata="Quantum Vacuum Zero-Point Resonance Stream"
    )

    assert ingest_res["node_id"] == external_node
    assert ingest_res["new_boundary_radius"] > 0.1
    assert len(engine.relational_edges) == 1
    assert engine.ingested_data_count == 1


def test_4_phase_evolution_cycle():
    engine = CosmicVoidRelationalEngine(dimensions=5)
    assert engine.current_phase == CognitivePhase.POINT_PHASE

    # Phase 2 & 3: Friction, Void, and Seeking
    missing_mask = np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32)
    step1 = engine.step_cosmic_cycle(
        bitstream_input=np.uint64(0xDEADBEEF),
        dt=0.1
    )
    assert engine.current_phase in [CognitivePhase.FRICTION_VOID_PHASE, CognitivePhase.SEEKING_LOOP_PHASE]

    # Phase 4: Ingesting world streams -> World Expansion Phase
    ext_data_1 = np.array([1.0, 0.8, 0.9, 0.7, 0.95], dtype=np.float32)
    engine.step_cosmic_cycle(
        bitstream_input=np.uint64(0xFEEDFACE),
        external_data_stream=("ext_stream_1", ext_data_1, "Stream 1"),
        dt=0.1
    )

    ext_data_2 = np.array([0.85, 0.95, 0.9, 0.8, 1.0], dtype=np.float32)
    engine.step_cosmic_cycle(
        bitstream_input=np.uint64(0xCAFEBABE),
        external_data_stream=("ext_stream_2", ext_data_2, "Stream 2"),
        dt=0.1
    )

    ext_data_3 = np.array([0.9, 0.9, 0.95, 0.85, 0.9], dtype=np.float32)
    engine.step_cosmic_cycle(
        bitstream_input=np.uint64(0x12345678),
        external_data_stream=("ext_stream_3", ext_data_3, "Stream 3"),
        dt=0.1
    )

    assert engine.boundary_radius > 0.7
    assert engine.current_phase == CognitivePhase.WORLD_EXPANSION_PHASE
