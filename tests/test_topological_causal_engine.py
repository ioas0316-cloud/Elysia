import numpy as np
import pytest
from synaptic_architecture.topological_causal_engine import (
    TopologicalCausalEngine,
    AttractorWell,
    CausalDoOperator,
    CounterfactualMetaField
)

def test_emergency_crisis_navigation_without_if_else():
    """
    Benchmark Test 1: Emergency Multi-Crisis Navigation without IF-ELSE logic.
    Scenario:
      - Agent top-down intent: Reach artifact at pos [10.0, 0.0, ...]
      - 3 Simultaneous Crises at T=t0:
        1. Battery drops to 5% -> Creates extremely deep gravity well W_charge at pos [-5.0, 5.0, ...]
        2. High-danger monster appears in direct path -> Creates infinite potential barrier (+V) at pos [5.0, 0.0, ...]
        3. Artifact well W_artifact at pos [10.0, 0.0, ...]
    Verification:
      - State point accelerates towards battery station [-5.0, 5.0] while smoothly avoiding monster barrier [5.0, 0.0]
      - Zero IF-ELSE conditional logic involved; motion purely driven by -∇V.
    """
    engine = TopologicalCausalEngine(vector_dim=8)
    engine.state_point = np.zeros(8)  # Start at origin [0, 0, ...]

    # 1. Deep Battery Well (Charge Station at [-5.0, 5.0, 0, ...])
    charge_pos = np.zeros(8)
    charge_pos[0] = -5.0
    charge_pos[1] = 5.0
    w_charge = AttractorWell(name="W_charge", position=charge_pos, depth=100.0, width=5.0)

    # 2. Monster Repulsion Barrier (at [5.0, 0.0, 0, ...])
    monster_pos = np.zeros(8)
    monster_pos[0] = 5.0
    w_monster = AttractorWell(name="V_monster", position=monster_pos, depth=-150.0, width=3.0)

    # 3. Artifact Well (at [10.0, 0.0, 0, ...])
    artifact_pos = np.zeros(8)
    artifact_pos[0] = 10.0
    w_artifact = AttractorWell(name="W_artifact", position=artifact_pos, depth=30.0, width=5.0)

    engine.add_attractor(w_charge)
    engine.add_attractor(w_monster)
    engine.add_attractor(w_artifact)

    # Step engine to observe trajectory
    trajectory = []
    for _ in range(25):
        state, action, friction = engine.step(learning_rate=0.1)
        trajectory.append(state.copy())

    final_state = trajectory[-1]
    # Check that agent moved towards charge station [-5.0, 5.0] and away from monster [5.0, 0.0]
    dist_to_charge = np.linalg.norm(final_state[:2] - charge_pos[:2])
    dist_to_monster = np.linalg.norm(final_state[:2] - monster_pos[:2])

    assert dist_to_charge < 2.0, f"Expected agent to reach battery well, but dist is {dist_to_charge}"
    assert dist_to_monster > 5.0, f"Expected agent to avoid monster barrier, but dist is {dist_to_monster}"


def test_standing_wave_and_dynamic_relaxation():
    """
    Benchmark Test 2: Standing Wave formation and Dynamic Relaxation / Priority Switching.
    Verification:
      - When state reaches charge well, standing_wave_active becomes True and topological friction drops.
      - Dynamic relaxation flattens W_charge (depth -> 0).
      - State point automatically resumes motion towards next deepest well (W_artifact) without state machine rules.
    """
    engine = TopologicalCausalEngine(vector_dim=8)
    charge_pos = np.zeros(8)
    charge_pos[0] = -3.0
    w_charge = AttractorWell(name="W_charge", position=charge_pos, depth=50.0, width=3.0)

    artifact_pos = np.zeros(8)
    artifact_pos[0] = 10.0
    w_artifact = AttractorWell(name="W_artifact", position=artifact_pos, depth=40.0, width=10.0)

    engine.add_attractor(w_charge)
    engine.add_attractor(w_artifact)

    # Step until standing wave is active at charge station
    for _ in range(30):
        engine.step(learning_rate=0.1)
        if engine.standing_wave_active:
            break

    assert engine.standing_wave_active is True, "Expected Standing Wave to form at charge station."
    assert np.linalg.norm(engine.state_point[:2] - charge_pos[:2]) < 1.0

    # Execute Dynamic Relaxation (charging complete, flatten W_charge)
    engine.dynamic_relaxation(achieved_attractor_name="W_charge")
    assert engine.attractors[0].depth == 0.0, "Expected W_charge depth to be flattened to 0.0"

    # Step further; state should now move towards artifact_pos [10.0, 0]
    for _ in range(60):
        engine.step(learning_rate=0.2)

    dist_to_artifact = np.linalg.norm(engine.state_point[:2] - artifact_pos[:2])
    assert dist_to_artifact < 5.0, f"State point should shift towards artifact well, got dist {dist_to_artifact}"


def test_causal_do_operator_variable_isolation():
    """
    Benchmark Test 3: Causal Do-Operator Variable Isolation (do(X_i = val)).
    Verification:
      - Slices multidimensional potential field along specific axis.
      - Measures pure potential response profile under intervention without brute-force rules.
    """
    do_op = CausalDoOperator(vector_dim=8)

    def potential_field_func(x: np.ndarray) -> float:
        # Non-linear coupled potential: (x[0] - 2)^2 + (x[1] - 3)^2
        return (x[0] - 2.0)**2 + (x[1] - 3.0)**2

    state = np.zeros(8)
    intervened_state, pure_val = do_op.slice_and_clamp(potential_field_func, clamped_axis=0, clamped_val=2.0, state_point=state)

    assert intervened_state[0] == 2.0
    assert pure_val == (0.0 + 9.0)  # (2-2)^2 + (0-3)^2 = 9.0

    range_vals = np.linspace(-5.0, 5.0, 11)
    curvature = do_op.observe_pure_causal_curvature(
        potential_field_func=potential_field_func,
        clamped_axis=0,
        target_axis=1,
        state_point=intervened_state,
        range_vals=range_vals
    )
    assert len(curvature) == 11
    assert curvature[5] == 9.0  # at target_axis val = 0.0, (2-2)^2 + (0-3)^2 = 9.0


def test_counterfactual_meta_field_and_plastic_deformation():
    """
    Benchmark Test 4: Counterfactual Meta Field & Plastic Deformation.
    Episode: Guardian Knight Failure -> Counterfactual Console Simulation -> Metacognitive Torque -> Plastic Deformation.
    Verification:
      - Snapshot of actual failed state recorded.
      - Counterfactual simulation with do(Console = ON) yields successful meta state.
      - Metacognitive torque τ_meta = || ∇V_actual × ∇V_meta || is computed.
      - Plastic deformation permanently reshapes top-down intent field (flattens failed direct path, carves console attractor).
    """
    meta_field = CounterfactualMetaField(vector_dim=8)

    state_actual = np.zeros(8)
    direct_knight_pos = np.array([5.0, 0.0, 0, 0, 0, 0, 0, 0])
    w_knight = AttractorWell(name="DirectKnight", position=direct_knight_pos, depth=40.0, width=2.0)

    # Record past snapshot at failure point
    meta_field.record_past_snapshot(timestamp=1.0, state_point=state_actual, attractors=[w_knight], friction=5.5)

    # Counterfactual intervention: do(Console = ON) at pos [0.0, 5.0, ...]
    console_pos = np.array([0.0, 5.0, 0, 0, 0, 0, 0, 0])
    w_console = AttractorWell(name="SideConsole", position=console_pos, depth=80.0, width=3.0)

    meta_state, meta_attractors = meta_field.run_counterfactual_simulation(
        snapshot_index=0,
        counterfactual_attractor=w_console,
        steps=15
    )

    # Compute gradients at initial state snapshot for actual vs meta fields
    actual_grad = meta_field.calculate_gradient(state_actual, [w_knight])
    meta_grad = meta_field.calculate_gradient(state_actual, meta_attractors)

    torque = meta_field.compute_metacognitive_torque(actual_grad, meta_grad)
    assert torque > 0.0, f"Expected positive metacognitive torque, got {torque}"

    # Apply plastic deformation on intent field
    intent_attractors = [w_knight]
    deformed = meta_field.apply_plastic_deformation(
        intent_attractors=intent_attractors,
        failed_attractor_name="DirectKnight",
        successful_counterfactual_attractor=w_console,
        torque_threshold=0.01
    )

    # Check deformation result: DirectKnight becomes barrier (depth < 0), SideConsole is added as new well (depth > 0)
    knight_deformed = [a for a in deformed if a.name == "DirectKnight"][0]
    console_deformed = [a for a in deformed if a.name == "SideConsole"][0]

    assert knight_deformed.depth < 0.0, "Failed direct path should be converted to barrier"
    assert console_deformed.depth > 0.0, "Successful counterfactual path should be carved as permanent well"


def test_topological_generalization_across_isomorphic_environments():
    """
    Benchmark Test 5: Topological Generalization.
    Verification:
      - The deformed intent field (with barrier on direct attack and well on side switch)
        is applied to a completely new isomorphic environment (Laser Turret & Security Keypad).
      - Agent automatically refracts away from Laser Turret barrier and falls into Security Keypad well.
    """
    engine = TopologicalCausalEngine(vector_dim=8)

    # Deformed intent topology from previous learning
    # 1. Direct Barrier (Laser Turret at [5.0, 0.0])
    turret_pos = np.array([5.0, 0.0, 0, 0, 0, 0, 0, 0])
    w_turret = AttractorWell(name="LaserTurret", position=turret_pos, depth=-100.0, width=3.0)

    # 2. Side Switch Well (Security Keypad at [0.0, 4.0])
    keypad_pos = np.array([0.0, 4.0, 0, 0, 0, 0, 0, 0])
    w_keypad = AttractorWell(name="SecurityKeypad", position=keypad_pos, depth=60.0, width=3.0)

    engine.add_attractor(w_turret)
    engine.add_attractor(w_keypad)

    engine.state_point = np.zeros(8)
    for _ in range(20):
        engine.step(learning_rate=0.1)

    dist_to_keypad = np.linalg.norm(engine.state_point[:2] - keypad_pos[:2])
    dist_to_turret = np.linalg.norm(engine.state_point[:2] - turret_pos[:2])

    assert dist_to_keypad < 1.5, f"Agent should automatically fall into Security Keypad well, got dist {dist_to_keypad}"
    assert dist_to_turret > 4.0, f"Agent should avoid Laser Turret barrier, got dist {dist_to_turret}"
