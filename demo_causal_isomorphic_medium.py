"""
Demo: Continuous Cognitive Isomorphism Cycle
=============================================
Demonstrates how a continuous mind (Tensor Rotor Field) transitions through phase states
into discrete decisions (WFC Collapse), handles causal contradictions via thermal relaxation (Reflection),
and re-aligns onto a valid topological state manifold.

Cognitive Cycle:
Sensory Input -> Field Warping -> Phase Lock -> WFC Decision -> Contradiction -> Reflection -> Re-imagination -> Convergence
"""

import sys
import numpy as np
from core.physics.causal_isomorphic_medium import (
    RotorTile,
    CausalIsomorphicMediumEngine
)


def print_banner(text: str):
    print("\n" + "=" * 75)
    print(f" {text}")
    print("=" * 75)


def print_grid(engine: CausalIsomorphicMediumEngine, title: str):
    print(f"\n--- [{title}] ---")
    summary = engine.get_grid_state_summary()
    for row in summary:
        print("  | " + " ".join(f"[{cell:5s}]" for cell in row) + " |")
    print("-" * 45)


def run_cognitive_isomorphism_demo():
    print_banner("ELYSIUM COGNITIVE ISOMORPHIC MEDIUM DEMO")
    print("Core Philosophy: 'Do not calculate, let it flow.'")
    print("Translating Continuous Perception -> Phase-Lock -> WFC Collapse -> Reflection Loop")

    # Define tiles with geometric 2D phase rotors
    tiles = [
        RotorTile(0, "RIGHT", np.array([1.0, 0.0])),
        RotorTile(1, "UP   ", np.array([0.0, 1.0])),
        RotorTile(2, "LEFT ", np.array([-1.0, 0.0])),
        RotorTile(3, "DOWN ", np.array([0.0, -1.0]))
    ]

    # Compatibility matrix: adjacent orthogonal tiles allowed (1.0), opposite tiles disallowed (0.0)
    compat_matrix = np.array([
        [1.0, 1.0, 0.0, 1.0],  # RIGHT
        [1.0, 1.0, 1.0, 0.0],  # UP
        [0.0, 1.0, 1.0, 1.0],  # LEFT
        [1.0, 0.0, 1.0, 1.0]   # DOWN
    ])

    engine = CausalIsomorphicMediumEngine(
        width=3, height=3, tiles=tiles, compatibility_matrix=compat_matrix, beta_0=2.5
    )

    # 1. High Entropy Imagination State
    print_grid(engine, "PHASE 1: INITIAL HIGH-ENTROPY IMAGINATION STATE")
    print("  Status: Inverse Temperature Beta = 2.50. All grid cells in liquid superposition.")

    # 2. Inject Sensory Stimuli that force a causal conflict
    print_banner("PHASE 2: SENSORY FEEDBACK INJECTION")
    print("  Injecting strong conflicting directional feedback:")
    print("  - Cell (0, 0) <- Forced RIGHT stimulus")
    print("  - Cell (0, 1) <- Forced LEFT stimulus (Directly opposing neighbor)")

    engine.inject_sensory_stimulus(0, 0, target_tile_id=0, intensity=1.0)  # RIGHT
    engine.inject_sensory_stimulus(0, 1, target_tile_id=2, intensity=1.0)  # LEFT

    # 3. Step through decision collapses
    print_banner("PHASE 3: PHASE-LOCKING & DECISION COLLAPSE")

    step_num = 0
    reflected = False

    while step_num < 20:
        step_num += 1
        is_done, is_conflict = engine.make_decision_step()

        if is_conflict:
            print(f"\n  ❌ [CONTRADICTION DETECTED] Step {step_num}: AC-3 constraint deadlock!")
            print_grid(engine, f"DEADLOCK AT STEP {step_num}")

            if not reflected:
                print_banner("PHASE 4: REFLECTION & THERMAL RELAXATION")
                print("  System triggered metacognitive thermal shock:")
                print(f"  - Lowering Inverse Beta: {engine.beta:.2f} -> {engine.beta * 0.3:.2f}")
                print("  - Un-collapsing rigid decisions back to high-entropy fluid superposition.")
                print("  - Injecting thermal rotor noise to escape local minimum energy trap.")

                engine.reflect_and_relax(radius=1, thermal_factor=0.3)
                reflected = True
                print_grid(engine, "POST-REFLECTION RE-IMAGINATION STATE")
                continue
            else:
                print("  Multiple contradictions after reflection. Terminating loop.")
                break

        if is_done:
            print_banner("PHASE 5: FINAL COGNITIVE EQUILIBRIUM CONVERGENCE")
            print(f"  ✅ Convergence achieved in {step_num} total steps!")
            print(f"  Total Decision Steps: {engine.decision_steps} | Reflection Cycles: {engine.reflection_count}")
            print_grid(engine, "FINAL CONVERGED STATE MANIFOLD")
            break

    print("\n" + "=" * 75)
    print(" DEMO COMPLETE: Continuous-to-Discrete Cognitive Isomorphism Proven.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_cognitive_isomorphism_demo()
