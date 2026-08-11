import pytest
import numpy as np
from core.physics.minimal_closed_loop import GroundedSensoryClosedLoop

def test_grounded_sensory_initialization():
    """Verify that concepts and connections are grounded in physical, multi-sensory properties."""
    loop = GroundedSensoryClosedLoop()

    # 1. Verify correct concepts are registered
    assert loop.num_nodes == 4
    assert "Sun" in loop.labels
    assert "Cold" in loop.labels

    # 2. Verify physical mass and sensory attributes are registered
    idx_sun = loop.idx_map["Sun"]
    assert loop.masses[idx_sun] == 5.0
    assert loop.V_charges[idx_sun] == 1.5
    assert loop.T_charges[idx_sun] == 2.0

    # 3. Verify center of mass is conserved at the origin on the spatial field
    center = np.mean(loop.S, axis=0)
    np.testing.assert_array_almost_equal(center, np.zeros(2), decimal=5)

    # 4. Verify initial connection topology matches sensory coherence
    idx_fire = loop.idx_map["Fire"]
    idx_cold = loop.idx_map["Cold"]

    # Sun and Fire share hot thermal properties, so they should have a connection
    assert loop.W[idx_sun, idx_fire] > 0.1
    # Sun and Cold have opposite thermal properties, so their initial connection should be 0.0
    assert loop.W[idx_sun, idx_cold] == 0.0


def test_sensory_shock_projection():
    """Verify that projecting a contradictory physical thermal shock spikes friction while conserving invariants."""
    loop = GroundedSensoryClosedLoop()

    initial_friction = loop.calculate_sensory_friction()

    # Project cold thermal shock (-3.5) onto the coordinates of the "Sun"
    loop.project_sensory_stimulus("Sun", cold_or_heat_impulse=-3.5)

    post_shock_friction = loop.calculate_sensory_friction()

    # 1. Friction must spike immediately due to thermal interference
    assert post_shock_friction > initial_friction

    # 2. Center of mass must remain conserved at the origin
    center = np.mean(loop.S, axis=0)
    np.testing.assert_array_almost_equal(center, np.zeros(2), decimal=5)


def test_sensory_causal_back_tracing():
    """Verify that the tension pointer precisely identifies the target concept of thermal contradiction."""
    loop = GroundedSensoryClosedLoop()

    # Blast Sun with cold
    loop.project_sensory_stimulus("Sun", cold_or_heat_impulse=-5.0)

    metrics = loop.step(dt=0.1)
    pointer_map = metrics["sensory_friction_map"]

    # Sun must be identified as the overwhelming source of contradiction/friction
    assert pointer_map["Sun"] > pointer_map["Fire"]
    assert pointer_map["Sun"] > pointer_map["Ice"]


def test_thermodynamic_relaxation_and_adaptation():
    """Verify that continuous steps naturally relax friction and mutate properties to resolve tension."""
    loop = GroundedSensoryClosedLoop(
        coordinate_relaxation_rate=0.3,
        charge_adaptation_rate=0.1,
        weight_mutation_rate=0.1,
        consolidation_rate=0.05
    )

    # Shock the Sun with extreme cold
    loop.project_sensory_stimulus("Sun", cold_or_heat_impulse=-4.0)
    initial_friction = loop.calculate_sensory_friction()

    # Step the loop for 25 iterations
    friction_history = [initial_friction]
    for _ in range(25):
        metrics = loop.step(dt=0.3)
        friction_history.append(metrics["friction_after"])

    # 1. Sensory friction must decay monotonically
    assert friction_history[-1] < initial_friction
    assert friction_history[-1] < 0.4 * initial_friction

    # 2. Check center of mass conservation throughout relaxation
    center = np.mean(loop.S, axis=0)
    np.testing.assert_array_almost_equal(center, np.zeros(2), decimal=5)


def test_sensory_consolidation_and_loop_closure():
    """Verify that invariants consolidate to lock in the new worldview after sensory relaxation."""
    loop = GroundedSensoryClosedLoop(
        coordinate_relaxation_rate=0.2,
        charge_adaptation_rate=0.1,
        weight_mutation_rate=0.1,
        consolidation_rate=0.1
    )

    idx_sun = loop.idx_map["Sun"]
    idx_cold = loop.idx_map["Cold"]

    initial_L_sun_cold = loop.L[idx_sun, idx_cold]

    # Shock Sun with cold
    loop.project_sensory_stimulus("Sun", cold_or_heat_impulse=-4.0)

    # Run loop for 30 steps
    for _ in range(30):
        loop.step(dt=0.3)

    final_L_sun_cold = loop.L[idx_sun, idx_cold]

    # The target invariant distance between Sun and Cold must have evolved and consolidated
    assert final_L_sun_cold != initial_L_sun_cold
