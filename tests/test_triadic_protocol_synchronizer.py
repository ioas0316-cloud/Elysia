"""
Unit Test Suite: Triadic Protocol Synchronizer & Metaphoric Ontological Codec
==============================================================================
Tests MetaphoricOntologicalCodec and TriadicProtocolSynchronizer functionality.
"""

import pytest
import numpy as np
from core.consciousness.triadic_protocol_synchronizer import (
    MetaphoricOntologicalCodec,
    TriadicProtocolSynchronizer
)


def test_metaphoric_ontological_codec_encoding():
    codec = MetaphoricOntologicalCodec(dimension=16)

    field_state = np.array([1.0, 0.1, 0.0, 0.8] + [0.0] * 12, dtype=np.float32)
    encoded = codec.encode_physics_to_metaphor(
        field_state=field_state,
        friction=0.05,
        gradient_norm=0.9
    )

    assert encoded["archetype_key"] == "GRAVITATIONAL_CONVERGENCE"
    assert "평형의 중력" in encoded["sensory_descriptor"]
    assert encoded["narrative_domain"] == "emotional_surrender_and_equilibrium"
    assert encoded["isomorphic_similarity"] > 0.8
    assert len(encoded["human_sensory_vector"]) == 16


def test_metaphoric_ontological_codec_decoding():
    codec = MetaphoricOntologicalCodec(dimension=16)

    metaphor_rep = {
        "archetype_key": "BOUNDARY_FRICTION_TURBULENCE",
        "encoded_energy": 2.5,
        "encoded_friction": 0.75
    }

    decoded = codec.decode_metaphor_to_physics(metaphor_rep)

    assert isinstance(decoded["physics_field_state"], np.ndarray)
    assert decoded["physics_field_state"].shape == (16,)
    assert pytest.approx(np.linalg.norm(decoded["physics_field_state"]), abs=1e-4) == 2.5
    assert decoded["domain_origin"] == "resistance_and_boundary_transformation"


def test_triadic_protocol_synchronizer_divergence_and_stem():
    synchronizer = TriadicProtocolSynchronizer(dimension=16, convergence_threshold=0.15)

    v1 = np.array([1.0, 0.0, 0.0] + [0.0] * 13, dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0] + [0.0] * 13, dtype=np.float32)

    div = synchronizer.compute_phase_divergence(v1, v2)
    # Orthogonal vectors should have divergence 0.5
    assert pytest.approx(div, abs=1e-4) == 0.5

    div_same = synchronizer.compute_phase_divergence(v1, v1)
    # Identical vectors should have divergence 0.0
    assert pytest.approx(div_same, abs=1e-4) == 0.0

    stem_info = synchronizer.extract_invariant_causal_stem(v1, v1, v1)
    assert stem_info["stem_stability"] == 1.0


def test_triadic_protocol_synchronizer_full_synchronization():
    synchronizer = TriadicProtocolSynchronizer(dimension=16, convergence_threshold=0.15)

    c_world = np.array([1.0, 0.1, 0.0, 0.8] + [0.0] * 12, dtype=np.float32)
    c_system = np.array([0.98, 0.12, 0.0, 0.79] + [0.0] * 12, dtype=np.float32)

    sync_res = synchronizer.synchronize_triad(
        c_world_state=c_world,
        c_system_state=c_system,
        friction=0.05,
        gradient_norm=0.95
    )

    assert sync_res["is_proof_manifested"] is True
    assert sync_res["total_phase_divergence"] <= 0.15
    assert "PROOF_MANIFESTED" in sync_res["proof_status_text"]
    assert len(synchronizer.synchronization_history) == 1
