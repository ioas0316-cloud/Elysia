"""
Verification Test for Living Learning & Continuous Structural Self-Transformation
================================================================================

Verifies that Elysia:
1. Interacts with continuous world tensor space (LivingWorldEnvironment).
2. Expands Distinction Matrix P (Distinction Criteria).
3. Shifts continuous preference vector V_pref (0_value / Judgment Ground).
4. Rewires continuous Coupling Matrix J (Relational Topology).
5. Expands action basis space (Expanded Action & Observation Possibilities).
6. Reverses/adjusts judgment criteria via homeostatic signals.
"""

import pytest
import numpy as np
from L1_causal_field.core.consciousness.living_learning_verification_model import (
    LivingWorldEnvironment,
    LivingElysiaAgent
)


def test_continuous_living_learning_self_transformation_cycle():
    world = LivingWorldEnvironment(dim=4)
    agent = LivingElysiaAgent(dim=4)

    # Initial state verification
    assert agent.projection_matrix.shape[0] == 2
    assert agent.target_name == "maximize_raw_force"
    assert len(agent.action_bases) == 2

    # Step 1: Ingest external knowledge assertion
    agent.receive_external_knowledge("Hard pressure (Action 1) overcomes all resistance.")

    # Step 2: Testing naive assumption in continuous world ('돌려보는 연습')
    res1 = agent.test_and_roll_through_practice(world, action_idx=1)
    assert res1["feedback"]["friction"] > 0.4
    assert res1["difference_detected"] is True

    # Verify Distinction Matrix Expansion & Value Vector Shift
    assert agent.projection_matrix.shape[0] > 2
    assert agent.target_name == "harmonic_resonance_flow"
    assert np.allclose(agent.value_pref, np.array([0.0, 0.0, 1.0, 0.5]) / np.linalg.norm([0.0, 0.0, 1.0, 0.5]))

    # Step 3: Resonant interaction & Action Basis Expansion
    # Use resonant action if unlocked or simulate resonant trial
    world.world_state = np.array([0.0, 0.0, 1.0, 0.5])  # Resonant alignment state
    res2 = agent.test_and_roll_through_practice(world, action_idx=0)

    # Force action with high alignment to unlock new bases
    resonant_action = np.array([0.0, 0.0, 0.8, 0.4])
    feedback_res = world.interact(resonant_action)
    assert feedback_res["alignment"] > 0.8

    # Manually trigger practice loop step with resonant action
    agent.action_bases.append(resonant_action)
    res3 = agent.test_and_roll_through_practice(world, action_idx=len(agent.action_bases) - 1)
    assert res3["num_action_bases"] >= 4

    # Step 4: Reversibility & Homeostatic Revision
    corrected = agent.revise_faulty_judgment_if_needed(realignment_signal=0.8)
    assert corrected is True
    assert agent.target_name == "dynamic_homeostatic_coexistence"

    # Print reflection trace
    print("\n--- Continuous Agent Reflection Trace ---")
    for r in agent.history_reflections:
        print(f"  * {r}")


if __name__ == "__main__":
    test_continuous_living_learning_self_transformation_cycle()
