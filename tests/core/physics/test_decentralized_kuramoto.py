"""
Unit Tests: Decentralized High-Dimensional Kuramoto Phase Synchronization Engine
=============================================================================
Tests global phase convergence, rogue phase attack autopoietic edge cutting,
abrupt node crash topology re-normalization, Meta-Rotor background curvature
gravitational reconnection, and SPDK live stream locking.
"""

import pytest
import math
from core.physics.decentralized_kuramoto_engine import (
    Vector3,
    Bivector3,
    Rotor3,
    DecentralizedKuramotoEngine
)


def test_clifford_rotor_and_vector_primitives():
    v1 = Vector3(1.0, 0.0, 0.0)
    v2 = Vector3(0.0, 1.0, 0.0)

    assert abs(v1.length() - 1.0) < 1e-6
    assert abs(v1.dot(v2)) < 1e-6

    # Wedge product anti-commutativity
    w12 = Bivector3.wedge(v1, v2)
    w21 = Bivector3.wedge(v2, v1)
    assert abs(w12.xy - 1.0) < 1e-6
    assert abs(w12.xy + w21.xy) < 1e-6

    # Rotor 90-degree rotation in e12 plane
    rotor = Rotor3.from_incremental_bivector(Bivector3(math.pi / 2.0, 0.0, 0.0))
    v_rot = rotor.rotate(v1)
    assert abs(v_rot.length() - 1.0) < 1e-5


def test_phase_synchronization_convergence():
    engine = DecentralizedKuramotoEngine(num_nodes=5, coupling_gain_k=8.0, seed=42)
    initial_z = engine.compute_global_coherence()

    # Step simulation for 50 ticks
    for _ in range(50):
        engine.step(dt=0.02)

    final_z = engine.compute_global_coherence()
    assert final_z > initial_z
    assert final_z > 0.95


def test_rogue_phase_attack_autopoietic_isolation():
    engine = DecentralizedKuramotoEngine(num_nodes=5, coupling_gain_k=8.0, phi_crit=0.85, seed=1337)

    # Align network initially
    for _ in range(40):
        engine.step(dt=0.02)

    assert engine.compute_global_coherence() > 0.95
    initial_edges = engine.get_active_edge_count()

    # Inject rogue attack on Node 2
    engine.inject_rogue_attack(2)

    for _ in range(15):
        engine.step(dt=0.02)

    # Active edges connected to Node 2 should be severed due to phi_crit violation
    active_edges_after_attack = engine.get_active_edge_count()
    assert active_edges_after_attack < initial_edges

    # Healthy nodes maintain high phase coherence
    healthy_z = engine.compute_global_coherence()
    assert healthy_z > 0.90


def test_abrupt_node_crash_topology_renormalization():
    engine = DecentralizedKuramotoEngine(num_nodes=5, seed=123)

    for _ in range(30):
        engine.step(dt=0.02)

    # Crash Node 0
    engine.kill_node(0)
    assert engine.nodes[0].is_dead is True

    # Run remaining steps
    for _ in range(20):
        engine.step(dt=0.02)

    # Remaining 4 active nodes maintain coherence
    summary = engine.get_topology_summary()
    assert summary["global_coherence_z"] > 0.90


def test_meta_rotor_gravitational_reconnection():
    engine = DecentralizedKuramotoEngine(num_nodes=5, coupling_gain_k=10.0, phi_crit=0.85, gamma_meta=2.0, seed=99)

    # Align initial network
    for _ in range(40):
        engine.step(dt=0.02)

    # Attack and isolate Node 2
    engine.inject_rogue_attack(2)
    for _ in range(15):
        engine.step(dt=0.02)

    assert engine.nodes[2].is_rogue is True

    # Heal Node 2 -> Meta-Rotor pulls Node 2 back into orbital alignment with Z(t)
    engine.heal_node(2)
    for _ in range(40):
        engine.step(dt=0.02)

    # Node 2 should reconnect and global coherence should recover to near 1.0
    summary = engine.get_topology_summary()
    assert summary["global_coherence_z"] > 0.95
    assert not engine.nodes[2].is_isolated


def test_spdk_live_world_streaming_phase_lock():
    engine = DecentralizedKuramotoEngine(num_nodes=3, seed=777)
    node_id = 0
    stream_data = b"ELYSIA_LIVE_WORLD_STREAM_TOKEN_2026"

    o_target = engine.project_stream_bytes_to_sphere(stream_data)

    # Stream packet ticks to Node 0
    for _ in range(30):
        engine.process_live_stream_packet(node_id=node_id, packet_bytes=stream_data, dt=0.02)

    aligned_psi = engine.nodes[node_id].psi
    alignment_score = aligned_psi.dot(o_target)

    # Node's phase state should lock onto the streamed wave (cosine similarity > 0.95)
    assert alignment_score > 0.95
