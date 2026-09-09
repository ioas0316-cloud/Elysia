import pytest
import causal_engine as ce
import math

def test_collective_manifold_initialization():
    manifold = ce.CollectiveManifold(3, 16)
    assert len(manifold.protocells) == 3
    assert manifold.system_dimension == 1
    assert manifold.collective_coherence == 0.0
    assert not manifold.dimension_spawned

    for i, cell in enumerate(manifold.protocells):
        assert cell.id == i
        assert cell.field.num_cells == 16
        assert cell.symbiotic_coupling == 0.0

def test_symbiotic_alignment_and_dimension_spawning():
    manifold = ce.CollectiveManifold(2, 20)

    # Configure initial signals for protocell 0 and 1
    cell_a = manifold.protocells[0]
    cell_b = manifold.protocells[1]

    # Initialize signal amplitude and gradient telos to create friction/deficit & macro potential
    for i in range(20):
        cell_a.field.signal_amplitude[i] = 10.0 + i
        cell_b.field.signal_amplitude[i] = 10.0 + i
        cell_a.field.gradient_telos[i] = 0.5
        cell_b.field.gradient_telos[i] = 0.5
        cell_a.field.macro_potential[i] = 0.5
        cell_b.field.macro_potential[i] = 0.5

    # Initially phases might differ slightly
    cell_a.self_identity_phase = 0.0
    cell_b.self_identity_phase = 0.1

    # Step dynamics repeatedly
    for _ in range(50):
        ce.step_collective_manifold_dynamics(manifold, coupling_rate=0.5, deficit_threshold=0.01, dt=0.1)

    # Check that inter-coherence built up and symbiotic coupling increased
    assert manifold.collective_coherence > 0.5
    assert manifold.protocells[0].symbiotic_coupling > 0.0
    assert manifold.protocells[1].symbiotic_coupling > 0.0

    # If coherence exceeds threshold (0.7), system_dimension should increase (N -> N+1)
    if manifold.collective_coherence > 0.7:
        assert manifold.dimension_spawned is True
        assert manifold.system_dimension == 2
        assert manifold.topological_volume >= 1.0

def test_friction_reduction_via_symbiosis():
    # Single isolated protocell vs symbiotic pair
    manifold_isolated = ce.CollectiveManifold(1, 20)
    cell_iso = manifold_isolated.protocells[0]
    for i in range(20):
        cell_iso.field.signal_amplitude[i] = 10.0
        cell_iso.field.gradient_telos[i] = 0.8

    for _ in range(30):
        ce.step_collective_manifold_dynamics(manifold_isolated, coupling_rate=0.0, deficit_threshold=0.01, dt=0.1)

    friction_isolated = sum(cell_iso.field.execution_friction)

    manifold_coupled = ce.CollectiveManifold(2, 20)
    cell_a = manifold_coupled.protocells[0]
    cell_b = manifold_coupled.protocells[1]
    cell_a.self_identity_phase = 0.0
    cell_b.self_identity_phase = 0.02
    for i in range(20):
        cell_a.field.signal_amplitude[i] = 10.0
        cell_b.field.signal_amplitude[i] = 10.0
        cell_a.field.gradient_telos[i] = 0.8
        cell_b.field.gradient_telos[i] = 0.8

    for _ in range(30):
        ce.step_collective_manifold_dynamics(manifold_coupled, coupling_rate=0.5, deficit_threshold=0.01, dt=0.1)

    friction_coupled = sum(cell_a.field.execution_friction)

    # Symbiotic coupling should reduce execution friction
    assert friction_coupled < friction_isolated
