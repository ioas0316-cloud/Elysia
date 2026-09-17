"""
Demo Cognitive Learning Engine
Demonstrates the Cognitive Learning Engine in a synthetic sensorimotor bootstrap environment:
1. Low-dimensional scalar signal stream with action-feedback loops.
2. Axiom 1: Transition recording (triplets: prev_ref, current_val, interval_velocity).
3. Axiom 2: Density emergence (path weight reinforcement on repetition).
4. Axiom 3: Rule self-review trigger on density threshold exceeding.
5. Axiom 4: Multi-channel continuous symbol grounding.
6. Phase Transition (Ice -> Water -> Gas) upon disturbance/energy injection.
7. Dual Mode: Forward forecasting & Reverse abductive goal pathway search.
8. Holonic unitization under selection pressure.
"""

import time
from synaptic_architecture.cognitive_learning_engine import (
    CognitiveLearningEngine,
    CognitiveLearningConfig,
    PhaseMode,
)


class SyntheticSensorimotorEnvironment:
    """
    Simulates a low-dimensional environment with clock ticks and action-feedback response.
    """

    def __init__(self):
        self.state = 10.0
        self.clock = 0.0

    def step(self, action: float = 0.0) -> float:
        self.clock += 0.1
        # Baseline cyclical drift + action feedback
        drift = 0.5 * (1 if int(self.clock * 10) % 2 == 0 else -0.5)
        self.state += drift + action
        return self.state


def run_demo():
    print("=" * 70)
    print("COGNITIVE LEARNING ENGINE DEMO")
    print("=" * 70)

    config = CognitiveLearningConfig(
        REEVAL_THRESHOLD_MULTIPLIER=2.5,
        ICE_TO_WATER_ENERGY=5.0,
        WATER_TO_GAS_ENERGY=20.0,
        REINFORCE_RATE=0.8,
    )
    engine = CognitiveLearningEngine(config=config)
    env = SyntheticSensorimotorEnvironment()

    sim_time = 1000.0

    print("\n--- PHASE 1: Normal Cyclical Stream (ICE Phase & Path Density Emergence) ---")
    for i in range(15):
        sim_time += 0.05  # Short intervals so temporal decay doesn't wipe grounding
        val = env.step(action=0.0)
        # Add grounded sensory labels intermittently
        labels = {"texture_smooth": 0.8, "frequency_hertz": 50.0} if i % 2 == 0 else None
        evt = engine.record_transition(val, timestamp=sim_time, external_labels=labels)
        print(f"[Tick {i:02d}] Val: {val:5.2f} | Vel: {evt.velocity:6.2f} | Phase: {engine.current_phase} | Energy: {engine.accumulated_energy:5.2f}")

    print("\n--- PHASE 2: High Delta Disturbance Injection (Phase Shift ICE -> WATER -> GAS) ---")
    for i in range(10):
        sim_time += 0.05
        # Inject large delta action
        disturbance = 5.0 if i % 2 == 0 else -4.0
        val = env.step(action=disturbance)
        evt = engine.record_transition(val, timestamp=sim_time)
        print(f"[Disturb {i:02d}] Val: {val:5.2f} | Vel: {evt.velocity:6.2f} | Phase: {engine.current_phase} | Energy: {engine.accumulated_energy:5.2f}")

    print("\n--- PHASE 3: Path Density & Axiom 3 Self-Modification Trigger ---")
    # Repeatedly traverse a fixed transition route
    for i in range(8):
        sim_time += 0.05
        val = 10.0 if i % 2 == 0 else 10.5
        engine.record_transition(val, timestamp=sim_time)

    print(f"Self-Modification Alerts Count: {len(engine.self_modification_alerts)}")
    for alert in engine.self_modification_alerts:
        print(f"  --> {alert['message']}")

    print("\n--- PHASE 4: Axiom 4 Symbol Grounding Verification ---")
    grounded_count = 0
    for source, targets in engine.network.items():
        for target, edge in targets.items():
            if edge.co_occurred_labels:
                grounded_count += 1
                print(f"Path [{source} -> {target}] Grounded Labels: {edge.co_occurred_labels}")
    print(f"Total grounded path edges: {grounded_count}")

    print("\n--- PHASE 5: Forward Forecasting (5.1 Mode) ---")
    current_node = engine.current_state_node
    forecast = engine.predict_forward(steps=3)
    print(f"Current State: {current_node}")
    print(f"Predicted Trajectory: {forecast}")

    print("\n--- PHASE 6: Reverse Abductive Goal Search (5.2 Mode) ---")
    if engine.network:
        # Search backward pathways to reach state S_10.5
        target_candidate = "S_10.5"
        reverse_paths = engine.search_reverse_abduction(target_node=target_candidate)
        print(f"Target Goal Node: {target_candidate}")
        print(f"Discovered Reverse Pathways: {reverse_paths}")

    print("\n--- PHASE 7: Holonic Meta-Observation & Unitization ---")
    print(f"Holonic Matrix Counts: {engine.holonic_matrix}")
    print(f"Stable Emergent Units: {engine.stable_units}")

    print("\n" + "=" * 70)
    print("DEMO EXECUTED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
