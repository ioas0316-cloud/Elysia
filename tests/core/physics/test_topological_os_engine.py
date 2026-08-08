import numpy as np
import pytest
from core.physics.topological_os_engine import TopologicalOSEngine

def test_toroidal_distance_wrap_around():
    """Verify that toroidal distance properly wraps around the grid boundary."""
    engine = TopologicalOSEngine(grid_shape=(10, 10))

    # Distance between (0, 0) and (9, 9) in a 10x10 torus grid should wrap around to (1, 1).
    # The shortest distance should be sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.4142
    dist = engine.compute_toroidal_distance((0, 0), (9, 9))
    assert pytest.approx(dist, 1e-4) == np.sqrt(2)

    # Distance between (2, 2) and (2, 8) in 10x10 wraps around to distance of 4 instead of 6.
    dist_2 = engine.compute_toroidal_distance((2, 2), (2, 8))
    assert dist_2 == 4.0

def test_initial_vacuum_ground_state():
    """Verify engine starts in the vacuum ground state where potential and energy are zero."""
    engine = TopologicalOSEngine(grid_shape=(8, 8))

    # All RNS residues should be initialized to ground vacuum state of 1
    assert np.all(engine.residues == 1)

    # Potential and energy should be completely zero
    assert np.all(engine.get_potential() == 0.0)
    assert np.all(engine.energy == 0.0)

def test_impulse_injection_phase_perturbation():
    """Verify that injecting an impulse perturbs the RNS residues and potential landscape."""
    engine = TopologicalOSEngine(grid_shape=(8, 8), primes=[5, 7])

    # Inject impulse at (3, 3)
    engine.inject_impulse(y=3, x=3, magnitude=10.0, importance=2.0)

    # Energy at (3, 3) should be increased
    assert engine.energy[3, 3] == 10.0

    # Potential at (3, 3) should be greater than 0 due to perturbed residues
    potential = engine.get_potential()
    assert potential[3, 3] > 0.0

    # Rest of the grid potential should remain unchanged (0.0)
    assert potential[0, 0] == 0.0

def test_langevin_thermodynamic_relaxation_and_cooling():
    """Verify Langevin relaxation and cooling schedule over steps."""
    engine = TopologicalOSEngine(grid_shape=(6, 6), initial_temp=1.0, cooling_rate=0.9)

    # Stimulate/Inject impulse to create a high potential and energy state
    engine.inject_impulse(y=2, x=2, magnitude=15.0, importance=3.0)
    initial_temp = engine.temperature

    # Take a physics step
    engine.step(0.1)

    # Temperature should cool down based on the cooling schedule
    assert engine.temperature < initial_temp

    # Energy should dissipate/flow across the torus grid
    assert np.any(engine.energy > 0)

    # Verify that residues relax back towards 1 over time
    for _ in range(20):
        engine.step(0.1)

    # Potential and RNS residues should eventually return to ground vacuum state 1
    assert np.all(engine.residues == 1)
    assert np.all(engine.get_potential() == 0.0)

def test_active_discernment_damping_and_gradient():
    """Verify active discernment filtering (Damping) and prioritization (Gradient)."""
    engine = TopologicalOSEngine(grid_shape=(8, 8), damping_factor=0.3)

    # 1. Gradient (Prioritizing): A highly important impulse should create a steeper potential gradient
    # Inject high importance vs low importance impulse
    engine.inject_impulse(y=1, x=1, magnitude=10.0, importance=4.0) # High priority
    high_potential = engine.get_potential()[1, 1]

    # Reset and test low importance
    engine_low = TopologicalOSEngine(grid_shape=(8, 8))
    engine_low.inject_impulse(y=1, x=1, magnitude=10.0, importance=1.0) # Low priority
    low_potential = engine_low.get_potential()[1, 1]

    # High importance must form a steeper, deeper potential well
    assert high_potential > low_potential

    # 2. Damping (Self-Refusal / Noise Filtering):
    # If we inject a random high-potential noise with no resonance (mismatched wave profile),
    # the damping factor should kick in and suppress the energy quickly.
    engine_noise = TopologicalOSEngine(grid_shape=(8, 8), damping_factor=0.4)
    # Force phase wave to have a flat or zero signature at (4, 4) to ensure mismatch
    engine_noise.phase_waves[4, 4] = 0.0

    # Inject high potential impulse with zero resonance
    engine_noise.inject_impulse(y=4, x=4, magnitude=20.0, importance=3.0, wave_signature=0.0)

    # Friction at that coordinate should increase and energy should quickly damp
    engine_noise.step(0.1)
    assert engine_noise.friction[4, 4] > 1.0 # friction increased
    assert engine_noise.energy[4, 4] < 20.0  # damped down
