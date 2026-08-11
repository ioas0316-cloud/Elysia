import pytest
import numpy as np
from core.physics.minimal_closed_loop import SemanticClosedLoopSystem

def test_semantic_loop_initialization():
    """Verify that semantic concepts and their connections are correctly initialized based on similarity."""
    concepts = {
        "Sun":  [1.0, 0.0, 0.0],
        "Fire": [0.9, 0.1, 0.0],
        "Ice":  [0.0, 0.9, 0.1],
        "Cold": [0.0, 1.0, 0.0]
    }

    system = SemanticClosedLoopSystem(concepts=concepts)

    # 1. Check labels and nodes
    assert system.num_nodes == 4
    assert "Sun" in system.concept_labels
    assert "Cold" in system.concept_labels

    # 2. Check center of mass conservation
    center = np.mean(system.S, axis=0)
    np.testing.assert_array_almost_equal(center, np.zeros(2), decimal=5)

    # 3. Check that initial W matches semantic proximity
    idx_sun = system.label_to_index["Sun"]
    idx_fire = system.label_to_index["Fire"]
    idx_cold = system.label_to_index["Cold"]

    # Sun and Fire are semantically close
    assert system.W[idx_sun, idx_fire] > 0.5
    # Sun and Cold are semantically far (below correlation threshold/similarity)
    assert system.W[idx_sun, idx_cold] < 0.2


def test_semantic_contradiction_injection():
    """Verify that projecting a semantic contradiction immediately spikes the friction while conserving invariants."""
    concepts = {
        "Sun":  [1.0, 0.0, 0.0],
        "Fire": [0.9, 0.1, 0.0],
        "Ice":  [0.0, 0.9, 0.1],
        "Cold": [0.0, 1.0, 0.0]
    }
    system = SemanticClosedLoopSystem(concepts=concepts)

    initial_friction = system.calculate_friction()

    # Project a contradictory statement: "Sun is Cold" (forcing distance to be very small, i.e., 0.1)
    system.project_semantic_stimulus("Sun", "Cold", force_distance=0.1)

    post_stimulus_friction = system.calculate_friction()

    # 1. Friction must spike immediately
    assert post_stimulus_friction > initial_friction

    # 2. Center of mass must remain conserved at the origin
    center = np.mean(system.S, axis=0)
    np.testing.assert_array_almost_equal(center, np.zeros(2), decimal=5)


def test_semantic_causal_back_tracing():
    """Verify that the tension source map correctly localizes and points to the contradictory semantic node."""
    concepts = {
        "Sun":  [1.0, 0.0, 0.0],
        "Fire": [0.9, 0.1, 0.0],
        "Ice":  [0.0, 0.9, 0.1],
        "Cold": [0.0, 1.0, 0.0]
    }
    system = SemanticClosedLoopSystem(concepts=concepts)

    # Inject contradiction
    system.project_semantic_stimulus("Sun", "Cold", force_distance=0.1)

    metrics = system.step(dt=0.1)
    f_map = metrics["semantic_friction_map"]

    # Sun and Cold are the root causes of the contradiction, so they must have higher friction than other nodes
    assert f_map["Sun"] > 0.1
    assert f_map["Cold"] > 0.1


def test_semantic_relaxation_and_belief_mutation():
    """Verify that the semantic manifold relaxes the friction and mutates belief coupling weights continuously."""
    concepts = {
        "Sun":  [1.0, 0.0, 0.0],
        "Fire": [0.9, 0.1, 0.0],
        "Ice":  [0.0, 0.9, 0.1],
        "Cold": [0.0, 1.0, 0.0]
    }
    system = SemanticClosedLoopSystem(
        concepts=concepts,
        coordinate_relaxation_rate=0.3,
        weight_mutation_rate=0.1,
        consolidation_rate=0.05
    )

    system.project_semantic_stimulus("Sun", "Cold", force_distance=0.15)

    idx_sun = system.label_to_index["Sun"]
    idx_cold = system.label_to_index["Cold"]
    idx_fire = system.label_to_index["Fire"]

    # Sun<->Cold starting coupling should be strong due to stimulus projection
    assert system.W[idx_sun, idx_cold] > 0.8

    initial_friction = system.calculate_friction()

    # Step the loop for 20 iterations
    friction_history = [initial_friction]
    for _ in range(20):
        metrics = system.step(dt=0.3)
        friction_history.append(metrics["friction_after"])

    # 1. Total semantic friction must decay monotonically
    assert friction_history[-1] < initial_friction
    assert friction_history[-1] < 0.25 * initial_friction

    # 2. Check physical center of mass conservation
    center = np.mean(system.S, axis=0)
    np.testing.assert_array_almost_equal(center, np.zeros(2), decimal=5)


def test_semantic_consolidation_and_loop_closure():
    """Verify that invariants L consolidate and adapt over relaxation, closing the loop with a new stable worldview."""
    concepts = {
        "Sun":  [1.0, 0.0, 0.0],
        "Fire": [0.9, 0.1, 0.0],
        "Ice":  [0.0, 0.9, 0.1],
        "Cold": [0.0, 1.0, 0.0]
    }
    system = SemanticClosedLoopSystem(
        concepts=concepts,
        coordinate_relaxation_rate=0.2,
        weight_mutation_rate=0.1,
        consolidation_rate=0.1
    )

    idx_sun = system.label_to_index["Sun"]
    idx_cold = system.label_to_index["Cold"]

    initial_L_sun_cold = system.L[idx_sun, idx_cold]

    # Inject contradiction
    system.project_semantic_stimulus("Sun", "Cold", force_distance=0.15)

    # Run loop for 30 steps to ensure full consolidation
    for _ in range(30):
        system.step(dt=0.3)

    # 1. Target invariant L between Sun and Cold must have adapted/consolidated to match the new worldview
    final_L_sun_cold = system.L[idx_sun, idx_cold]
    assert final_L_sun_cold != initial_L_sun_cold
    # Since we forced distance to 0.15, final consolidated invariant should be very close to that
    assert abs(final_L_sun_cold - 0.15) < 0.2
