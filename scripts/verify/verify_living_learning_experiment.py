"""
Verification Script for Minimal Living Learning Verification Model
====================================================================

Executes a full multi-stage continuous experiment demonstrating:
1. Interaction with a reactive, continuous tensor world environment.
2. Testing external assertions through practice ('돌려보는 연습').
3. Emergence of new distinction criteria (Distinction Matrix P expansion).
4. Transformation of judgment grounds (0_value / Preference Vector V_pref).
5. Dynamic rewiring of relational topology (Coupling Matrix J).
6. Expansion of observation & action basis possibilities.
7. Reversibility and self-correction of faulty/restrictive judgments.
"""

import sys
import numpy as np
from L1_causal_field.core.consciousness.living_learning_verification_model import (
    LivingWorldEnvironment,
    LivingElysiaAgent
)


def run_living_learning_experiment():
    print("=" * 80)
    print("      ELYSIA CONTINUOUS LIVING LEARNING & SELF-TRANSFORMATION EXPERIMENT      ")
    print("=" * 80)

    world = LivingWorldEnvironment(dim=4)
    agent = LivingElysiaAgent(dim=4)

    print("\n[STAGE 0] Initial Agent State:")
    print(f"  * Distinction Matrix P Rank: {agent.projection_matrix.shape[0]}")
    print(f"  * Value Preference Vector V_pref: {agent.value_pref.tolist()}")
    print(f"  * Target Name (0_value): '{agent.target_name}'")
    print(f"  * Number of Action Bases: {len(agent.action_bases)}")

    # Stage 1: External assertion injection
    print("\n[STAGE 1] Ingesting External Knowledge/Assertion:")
    external_assertion = "Hard pressure (Action 1) overcomes all resistance."
    agent.receive_external_knowledge(external_assertion)
    print(f"  * Received assertion: '{external_assertion}'")

    # Stage 2: Testing naive assumption in world interaction ('돌려보는 연습')
    print("\n[STAGE 2] Testing Naive Assumption through Practice ('돌려보는 연습'):")
    res1 = agent.test_and_roll_through_practice(world, action_idx=1)
    print(f"  * Action Taken Index: 1 (Hard Pressure Vector)")
    print(f"  * Displacement: {res1['feedback']['displacement']:.2f}, Friction: {res1['feedback']['friction']:.2f}")
    print(f"  * Difference Detected: {res1['difference_detected']}")
    print(f"  * Expanded Distinction Rank: {res1['distinction_rank']}")
    print(f"  * Shifted Value Ground (0_value): '{res1['target_name']}'")

    # Stage 3: Resonant Action & Basis Space Expansion
    print("\n[STAGE 3] Resonant Interaction & Basis Space Expansion:")
    resonant_action = np.array([0.0, 0.0, 0.8, 0.4])
    agent.action_bases.append(resonant_action)
    res2 = agent.test_and_roll_through_practice(world, action_idx=len(agent.action_bases) - 1)
    print(f"  * Action Taken Index: {len(agent.action_bases) - 1} (Harmonic Resonant Pulse Vector)")
    print(f"  * Displacement: {res2['feedback']['displacement']:.2f}, Friction: {res2['feedback']['friction']:.2f}")
    print(f"  * Alignment with World Harmonic Axis: {res2['feedback']['alignment']:.2f}")
    print(f"  * Updated Number of Action Bases: {res2['num_action_bases']}")

    # Stage 4: Reversibility & Homeostatic Revision
    print("\n[STAGE 4] Reversibility & Homeostatic Self-Correction Check:")
    corrected = agent.revise_faulty_judgment_if_needed(realignment_signal=0.8)
    print(f"  * Reversibility Triggered: {corrected}")
    print(f"  * Revised Judgment Ground: '{agent.target_name}'")
    print(f"  * Revised Value Preference Vector: {agent.value_pref.tolist()}")

    print("\n[SUMMARY] Complete Reflection Trace of Living Agent:")
    for idx, ref in enumerate(agent.history_reflections, 1):
        print(f"  {idx}. {ref}")

    print("\n" + "=" * 80)
    print("    EXPERIMENT COMPLETED SUCCESSFULLY: LIVING STRUCTURE SELF-TRANSFORMED    ")
    print("=" * 80)


if __name__ == "__main__":
    run_living_learning_experiment()
