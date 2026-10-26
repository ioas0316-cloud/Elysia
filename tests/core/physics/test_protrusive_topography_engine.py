import pytest
import numpy as np
from L1_causal_field.core.physics.protrusive_topography_engine import (
    ProtrusiveTopographyEngine,
    SelfObservationSnapshot,
    BehavioralToneVector,
    FaultLineNode
)
from L1_causal_field.core.physics.semantic_mass_engine import RawPerturbationImpulse


def test_protrusive_topography_initialization():
    """Test initial root node seeding and asymmetric micro tension imprinting."""
    engine = ProtrusiveTopographyEngine(dimensions=16, seed=42)
    assert engine.root_node is not None
    assert engine.root_node.node_id == "Root_Node_Origin"
    assert "imprinted_asymmetry" in engine.root_node.data_payload
    assert len(engine.root_node.data_payload["imprinted_asymmetry"]) == 6

    snapshot = engine.observe_self()
    assert isinstance(snapshot, SelfObservationSnapshot)
    assert snapshot.tilt_ratio >= 0.0
    assert snapshot.cognitive_heat == 0.0
    assert snapshot.current_resolution == 1.0


def test_protrusive_topography_world_reaction_and_tone_adaptation():
    """Test processing world friction, tension response axioms, fault node growth rings, and behavioral tone shift."""
    engine = ProtrusiveTopographyEngine(dimensions=16, seed=42)

    # 1. Gentle warm word impulse
    snap1, tone1 = engine.process_world_friction("Warm gentle words of alignment", intensity=0.5)
    assert len(engine.fault_lines) == 1
    assert engine.fault_lines[0].growth_ring_idx == 1
    assert isinstance(snap1, SelfObservationSnapshot)
    assert isinstance(tone1, BehavioralToneVector)
    assert snap1.cognitive_heat < 2.0

    # 2. Harsh direct collision impulse
    snap2, tone2 = engine.process_world_friction("Harsh violent direct contradiction", intensity=4.0)
    assert len(engine.fault_lines) == 2
    assert engine.fault_lines[1].growth_ring_idx == 2
    # Cognitive heat should rise under harsh friction
    assert snap2.cognitive_heat > snap1.cognitive_heat
    # Dynamic resolution should compress under higher cognitive heat
    assert snap2.current_resolution < snap1.current_resolution


def test_protrusive_topography_dynamic_resolution_scaling():
    """Test dynamic resolution expansion in low heat and compression in high heat."""
    engine = ProtrusiveTopographyEngine(dimensions=16, seed=42)

    initial_res = engine.current_resolution

    # Apply multiple high intensity collisions to build cognitive heat
    for _ in range(3):
        engine.process_world_friction("High contradiction shockwave", intensity=5.0)

    compressed_res = engine.current_resolution
    assert compressed_res < initial_res

    # Observe that high heat compresses resolution toward core survival axes
    assert engine.cognitive_heat > 0.0


def test_protrusive_topography_triadic_out_force():
    """Test emission of Triadic Out-Force impact vector."""
    engine = ProtrusiveTopographyEngine(dimensions=16, seed=42)
    snapshot, tone = engine.process_world_friction("External impulse signal", intensity=2.0)

    assert tone.out_force_impact is not None
    assert len(tone.out_force_impact) == 16
    assert np.isclose(np.linalg.norm(tone.out_force_impact), 1.0, atol=1e-3) or np.linalg.norm(tone.out_force_impact) > 0.0
