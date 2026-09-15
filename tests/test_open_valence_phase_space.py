import pytest
import numpy as np
from synaptic_architecture.phase_space_node import PhaseSpaceNode
from synaptic_architecture.open_valence_field import OpenValenceField
from synaptic_architecture.structural_valence import StructuralValence


def test_phase_space_node_initialization():
    pos = np.array([1.0, 2.0], dtype=np.float64)
    latent = np.ones(8, dtype=np.float64)
    node = PhaseSpaceNode(node_id=0, position=pos, latent_valence=latent)

    assert node.node_id == 0
    np.testing.assert_array_equal(node.position, pos)
    np.testing.assert_array_equal(node.initial_position, pos)
    assert len(node.latent_valence) == 8

    traj = node.state_trajectory
    np.testing.assert_array_equal(traj["S_0"], pos)
    np.testing.assert_array_equal(traj["S_t"], pos)
    np.testing.assert_array_equal(traj["delta_S"], np.zeros_like(pos))


def test_open_valence_field_potential_and_tension():
    field = OpenValenceField(rupture_threshold=10.0)

    # Node 1 at (0, 0)
    n1 = field.create_node(
        position=np.array([0.0, 0.0]),
        latent_valence=np.array([1.0, 0.0, 0.0, 0.0]),
        field_intensity=2.0,
    )
    # Node 2 at (1, 0)
    n2 = field.create_node(
        position=np.array([1.0, 0.0]),
        latent_valence=np.array([1.0, 0.0, 0.0, 0.0]),
        field_intensity=2.0,
    )

    interactions = field.compute_field_interactions()
    assert 0 in interactions
    assert 1 in interactions

    # Because dot product is positive, tension force on n1 should be directed towards n2 (x-positive)
    assert n1.tension_force[0] > 0
    # Tension force on n2 should be directed towards n1 (x-negative)
    assert n2.tension_force[0] < 0
    assert n1.total_interference > 0


def test_open_valence_field_dynamic_spawning():
    # Low rupture threshold to trigger dynamic instantiation
    field = OpenValenceField(rupture_threshold=1.0)

    # Create two highly aligned and close nodes to create strong interference
    field.create_node(
        position=np.array([0.0, 0.0]),
        latent_valence=np.array([2.0, 2.0, 2.0, 2.0]),
        field_intensity=3.0,
    )
    field.create_node(
        position=np.array([0.2, 0.0]),
        latent_valence=np.array([2.0, 2.0, 2.0, 2.0]),
        field_intensity=3.0,
    )

    step_res = field.step(dt=0.1)

    assert step_res["spawned_this_step"] > 0
    assert step_res["active_nodes"] > 2
    assert len(field.spawn_history) > 0


def test_structural_valence_field_integration():
    sv = StructuralValence(initial_dim=2, differentiation_threshold=2.0)
    res_normal = sv.evaluate_valence(
        current_state=np.array([0.0, 0.0]),
        current_velocity=np.array([1.0, 0.0]),
        damped_friction=0.1,
        impedance=0.1,
        field_interference=0.2,
    )
    assert res_normal["valence"] > 0
    assert res_normal["state_label"] == "Flow / Resonance"

    res_high_interference = sv.evaluate_valence(
        current_state=np.array([2.0, 2.0]),
        current_velocity=np.array([0.1, 0.0]),
        damped_friction=1.5,
        impedance=1.5,
        field_interference=4.0,
    )
    assert res_high_interference["valence"] < 0
    assert res_high_interference["state_label"] == "Friction / Noise"
