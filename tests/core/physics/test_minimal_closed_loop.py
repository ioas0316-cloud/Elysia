import pytest
import numpy as np
from core.physics.minimal_closed_loop import MinimalClosedLoopSystem

def test_initialization():
    """Verify that the Minimal Closed Loop System initializes at perfect, zero-friction rest."""
    num_nodes = 8
    dims = 2
    system = MinimalClosedLoopSystem(num_nodes=num_nodes, dimensions=dims)

    assert system.num_nodes == num_nodes
    assert system.dimensions == dims

    # 1. Verify State Space (S) dimensions
    assert system.S.shape == (num_nodes, dims)

    # 2. Verify physical conservation of center of mass (should be at origin)
    center_of_mass = np.mean(system.S, axis=0)
    np.testing.assert_array_almost_equal(center_of_mass, np.zeros(dims), decimal=5)

    # 3. Verify zero friction at rest state (F = 0)
    friction = system.calculate_friction()
    assert abs(friction) < 1e-5


def test_stimulus_projection_and_conservation():
    """Verify that external stimulus projects correctly and increases friction while conserving physical invariants."""
    system = MinimalClosedLoopSystem(num_nodes=8)

    # Apply a local perturbation to Node 3
    node_idx = 3
    impulse = np.array([1.2, -0.8], dtype=np.float32)

    system.project_stimulus(node_index=node_idx, impulse=impulse)

    # 1. Friction must rise immediately above 0
    perturbed_friction = system.calculate_friction()
    assert perturbed_friction > 0.1

    # 2. Center of mass must remain perfectly conserved at 0
    center_of_mass = np.mean(system.S, axis=0)
    np.testing.assert_array_almost_equal(center_of_mass, np.zeros(2), decimal=5)


def test_causal_back_tracing():
    """Verify that localized gradients correctly index/back-trace the source of friction."""
    system = MinimalClosedLoopSystem(num_nodes=8)

    # Apply a high-energy perturbation to Node 4
    node_idx = 4
    impulse = np.array([2.0, 1.5], dtype=np.float32)
    system.project_stimulus(node_index=node_idx, impulse=impulse)

    # Execute a step to capture the gradients
    metrics = system.step(dt=0.1)

    local_frictions = np.array(metrics["local_friction_index"])

    # The perturbed Node 4 and its circular neighbors (3 and 5) must bear the highest local friction (contradiction index)
    # Distant nodes (e.g. 0) should have significantly lower or zero initial friction gradients
    assert local_frictions[4] > local_frictions[0]
    assert local_frictions[3] > local_frictions[0]
    assert local_frictions[5] > local_frictions[0]


def test_autonomous_relaxation_and_convergence():
    """Verify that stepping the system causes friction to relax towards zero without hardcoded rules."""
    system = MinimalClosedLoopSystem(
        num_nodes=8,
        coordinate_relaxation_rate=0.3,
        weight_mutation_rate=0.05,
        consolidation_rate=0.05
    )

    # Perturb
    system.project_stimulus(node_index=2, impulse=np.array([1.5, -1.0], dtype=np.float32))
    initial_friction = system.calculate_friction()

    # Run the system for 40 steps
    friction_history = [initial_friction]
    for _ in range(40):
        metrics = system.step(dt=0.2)
        friction_history.append(metrics["friction_after"])

    # 1. Friction must show overall relaxation (decreasing trend)
    assert friction_history[-1] < initial_friction

    # 2. Friction should converge close to 0 (relaxed state)
    assert friction_history[-1] < 0.1 * initial_friction

    # 3. Center of mass must stay perfectly conserved at origin throughout relaxation
    center_of_mass = np.mean(system.S, axis=0)
    np.testing.assert_array_almost_equal(center_of_mass, np.zeros(2), decimal=5)


def test_topological_mutation_and_consolidation():
    """Verify that connection weights W and target lengths L continuously mutate and consolidate during the loop."""
    system = MinimalClosedLoopSystem(
        num_nodes=6,
        coordinate_relaxation_rate=0.2,
        weight_mutation_rate=0.1,
        consolidation_rate=0.1
    )

    # Capture initial states
    initial_W = system.W.copy()
    initial_L = system.L.copy()

    # Apply a massive perturbation to node 1 to induce strain
    system.project_stimulus(node_index=1, impulse=np.array([2.5, 2.5], dtype=np.float32))

    # Step multiple times to allow mutation and consolidation to propagate
    for _ in range(10):
        system.step(dt=0.2)

    # 1. W must have mutated (changed from initial weights)
    assert not np.array_equal(system.W, initial_W)

    # 2. L must have consolidated/adapted to accommodate the new structural shape
    assert not np.array_equal(system.L, initial_L)

    # 3. Weights should remain within physical bounds [0.1, 5.0] for active connections
    active_mask = system.initial_W > 0
    assert np.all(system.W[active_mask] >= 0.1)
    assert np.all(system.W[active_mask] <= 5.0)
