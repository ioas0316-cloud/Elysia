"""
Demonstration Script: Existential Meta-Cognition & Dynamic Intent Re-observation

This demo simulates a mental organism navigating 4D spacetime:
1. Initially navigating under "EFFICIENCY" intent phase along habit/scar trajectories.
2. Encounters sensory disruption triggering Existential Self-Query: "Is this path truly what I affirm?"
3. Re-observes past causal history and switches intent phase to "MEANING_RESONANCE".
4. Reconfigures metric tensor h_{ij}(x, t) and Christoffel symbols \\Gamma^\\mu_{\\alpha\\beta}.
5. Dynamically bends geodesic flow trajectory without backpropagation loss.
"""

import numpy as np
from core.consciousness.existential_meta_cognition_engine import ExistentialSelfQueryLoop


def run_existential_meta_cognition_demo():
    print("==========================================================================")
    print(" ELYSIA COGNITIVE ENGINE: EXISTENTIAL META-COGNITION & RE-OBSERVATION DEMO")
    print("==========================================================================\n")

    # Initialize 4D Spacetime Existential Self-Query Loop
    topological_dna = np.array([1.0, 0.5, 0.8, 0.3])
    loop = ExistentialSelfQueryLoop(dim=4, alpha=0.5, beta=2.5, topological_dna=topological_dna)

    # Initial state in 4D spacetime x^\mu = (t, x_1, x_2, x_3)
    pos = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    vel = np.array([1.0, 0.5, 0.2, 0.1], dtype=np.float64)
    sensory_wave = np.array([1.0, 0.5, 0.8, 0.3], dtype=np.float64)  # Perfectly aligned initially

    print("[STAGE 1] Navigating in 4D Spacetime under EFFICIENCY Intent Phase...")
    print(f"  - Initial Position: {pos}")
    print(f"  - Initial Velocity: {vel}")
    print(f"  - Active Phase:     {loop.compass.current_phase}")
    print(f"  - Attractor Coord:  {loop.compass.get_active_attractor()}")
    print()

    # Step forward 5 steps along EFFICIENCY geodesic flow
    for step in range(1, 6):
        record = loop.run_step(pos, vel, sensory_wave, dtau=0.08)
        pos = record["next_position"]
        vel = record["next_velocity"]
        print(f"  Step {step}: Pos={np.round(pos, 3)} | Vel={np.round(vel, 3)} | Friction={record['qualia_friction']:.3f}")

    print("\n--------------------------------------------------------------------------")
    print("[STAGE 2] Environmental Shock & Existential Self-Query Triggered")
    print("--------------------------------------------------------------------------")
    # Disruption in sensory wave causing qualia friction
    disrupted_wave = np.array([-2.0, 3.5, -1.0, 4.0], dtype=np.float64)

    print("  > Disrupted sensory wave received!")
    print("  > Self-Query: 'Is this habitual causal path truly aligned with my affirmed values?'")

    # Step with existential trigger enabled
    trigger_record = loop.run_step(
        pos,
        vel,
        disrupted_wave,
        dtau=0.08,
        existential_trigger=True,
        target_phase_on_trigger="MEANING_RESONANCE",
    )

    pos = trigger_record["next_position"]
    vel = trigger_record["next_velocity"]

    print(f"  - Existential Query Raised: {trigger_record['query_raised']}")
    print(f"  - Re-observation Occurred: {trigger_record['reobservation_occurred']}")
    print(f"  - Intent Phase Shift:      {trigger_record['previous_phase']} -> {trigger_record['current_phase']}")
    print(f"  - New Teleological Attractor: {trigger_record['active_attractor']}")
    print(f"  - Metric Tensor h_ij Reconfigured:\n{np.round(trigger_record['metric_tensor'], 3)}")
    print()

    print("--------------------------------------------------------------------------")
    print("[STAGE 3] Bending Geodesic Flow Trajectory under New Intent Compass")
    print("--------------------------------------------------------------------------")

    # Step forward 5 steps along MEANING_RESONANCE geodesic flow
    for step in range(7, 12):
        record = loop.run_step(pos, vel, sensory_wave, dtau=0.08)
        pos = record["next_position"]
        vel = record["next_velocity"]
        print(f"  Step {step}: Pos={np.round(pos, 3)} | Vel={np.round(vel, 3)} | Active Phase={record['current_phase']}")

    print("\n==========================================================================")
    print(" EXISTENTIAL META-COGNITION & RE-OBSERVATION DEMO COMPLETED SUCCESSFULLY.")
    print("==========================================================================")


if __name__ == "__main__":
    run_existential_meta_cognition_demo()
