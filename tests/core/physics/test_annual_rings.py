import pytest
import numpy as np
from core.physics.topological_os_engine import TopologicalOSEngine

def test_conductance_matrix_initialization():
    """Verify that conductance_matrix (Annual Rings) starts at uniform 1.0."""
    engine = TopologicalOSEngine(grid_shape=(8, 8))
    assert engine.conductance_matrix.shape == (8, 8)
    assert np.allclose(engine.conductance_matrix, 1.0)

def test_conductance_matrix_accumulation():
    """Verify that friction/energy dissipation causes permanent conductance increase (Annual Rings)."""
    engine = TopologicalOSEngine(grid_shape=(8, 8), initial_temp=0.0)

    # Inject a heavy impulse at (3, 3)
    engine.inject_impulse(y=3, x=3, magnitude=30.0, importance=5.0)

    initial_conductance = engine.conductance_matrix[3, 3]

    # Run steps to dissipate energy
    for _ in range(10):
        engine.step(0.1)

    final_conductance = engine.conductance_matrix[3, 3]

    # Conductance should have increased/etched permanently (나이테 축적)
    assert final_conductance > initial_conductance
    assert final_conductance > 1.0

def test_conductance_reduces_friction_and_accelerates_relaxation():
    """Verify that higher conductance (Annual Rings) leads to lower friction and faster energy/potential dissipation."""
    engine_fresh = TopologicalOSEngine(grid_shape=(8, 8), initial_temp=0.0)
    engine_etched = TopologicalOSEngine(grid_shape=(8, 8), initial_temp=0.0)

    # Pre-etch the etched engine by manually boosting local conductance around (4, 4)
    engine_etched.conductance_matrix[3:6, 3:6] = 4.0

    # Inject identical impulse to both engines
    engine_fresh.inject_impulse(y=4, x=4, magnitude=20.0, importance=4.0)
    engine_etched.inject_impulse(y=4, x=4, magnitude=20.0, importance=4.0)

    # Relax both for 15 steps
    for _ in range(15):
        engine_fresh.step(0.1)
        engine_etched.step(0.1)

    state_fresh = engine_fresh.get_state()
    state_etched = engine_etched.get_state()

    potential_fresh = np.sum(state_fresh["potential"])
    potential_etched = np.sum(state_etched["potential"])

    # The etched engine (with higher conductance / lower friction) should have relaxed closer to vacuum state 1
    assert potential_etched <= potential_fresh
