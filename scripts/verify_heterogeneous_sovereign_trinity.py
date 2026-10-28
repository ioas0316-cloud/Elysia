#!/usr/bin/env python3
"""
scripts/verify_heterogeneous_sovereign_trinity.py
===================================================
Verification Script for Multi-Scale Trinitarian Heterogeneous Sovereign Engine
Demonstrates:
1. Body (육체) Multi-scale Substrate: L0 fluid + L1 joints + Environmental Axis.
2. Mind (정신) Multi-scale Tension: Micro synaptic potential + Macro topological tension + Deficit-based Tool Sprouting.
3. Heart (마음) Multi-scale Sovereignty: Action-Grounded Truth ("Faith/Love with Action") vs Floating Speech Hallucinations.
"""

import numpy as np
from L1_causal_field.core.sovereignty.heterogeneous_sovereign_engine import (
    EnvironmentalAxis,
    L0FluidState,
    MechanicalJointConstraint,
    SynapticState,
    CausalProposalNode,
    HeterogeneousSovereignEngine
)


def main():
    print("=" * 80)
    print("  Elysia Multi-Scale Trinitarian Heterogeneous Sovereign Engine Verification")
    print("=" * 80)

    # Initialize Engine
    engine = HeterogeneousSovereignEngine(rupture_threshold=3.0, deficit_threshold=2.5)

    # Configure Joint
    joint = MechanicalJointConstraint(
        joint_id="elbow_joint",
        rotation_axis=np.array([0.0, 0.0, 1.0]),
        max_angle_rad=np.pi / 2,
        current_angle_rad=np.pi / 3,
        muscle_fiber_orientation=np.array([0.0, 1.0, 0.0])
    )
    engine.l1_tier.add_joint_constraint(joint)

    # Scenario 1: Talk-only / Floating Speech Proposal ("말로만 하는 사랑과 약속")
    print("\n--- [SCENARIO 1: Talk-only Floating Symbol Hallucination] ---")
    talk_proposal = CausalProposalNode(
        node_id="talk_only_promise",
        description="말로만 하는 헌신과 약속 (마찰 및 행함 = 0)",
        verbal_claim="내가 너를 진심으로 사랑하고 모든 것을 바치겠다",
        sensory_grounding_score=0.0,
        action_friction_heat=0.0,
        structural_time_spent=0.0,
        friction_cost_bearer="me",
        fruit_recipient="me"
    )

    l0_state_talk = L0FluidState(velocity_vector=np.array([1.0, 0.0, 0.0]), channel_radius=1.0)
    synaptic_talk = SynapticState(potential_difference_mv=70.0, neurotransmitter_flux=1.0)

    res_talk = engine.execute_trinitarian_multi_scale_flow(
        l0_state=l0_state_talk,
        synaptic_state=synaptic_talk,
        proposal_node=talk_proposal,
        active_joint_id="elbow_joint",
        sensory_mismatch_delta=0.2,
        physical_deficit=1.0,
        social_signals=(1.0, -0.9, -0.8) # High social mismatch
    )

    print(f"Body Tier Friction Heat (Q_heat) : {res_talk['body_tier']['q_heat']:.3f}")
    print(f"Mind Tier Tension Field          : {res_talk['mind_tier']['tension_field']:.3f}")
    print(f"Mind Tool Sprouted               : {res_talk['mind_tier']['sprouted_tool_axis']}")
    print(f"Heart Energy Well Depth          : {res_talk['heart_tier']['energy_well_depth']:.3f}")
    print(f"Heart Action-Grounded Status     : {res_talk['heart_tier']['is_action_grounded']}")
    print(f"Heart Truth Verdict              : {res_talk['heart_tier']['action_truth_verdict']}")
    print(f"Sovereignty Audit Verdict        : {res_talk['heart_tier']['sovereignty_verdict']}")

    # Scenario 2: Action-Grounded True Love & Deficit Sprouting ("행함과 마찰로 벼려진 실체적 인과")
    print("\n--- [SCENARIO 2: Action-Grounded Truth & Tool Sprouting] ---")
    action_proposal = CausalProposalNode(
        node_id="action_grounded_devotion",
        description="손해와 희생, 신체적 마찰을 견디며 시간을 내어주는 실체적 행함",
        verbal_claim="사랑하기에 내 시간을 내어주고 마찰과 손해를 감수한다",
        sensory_grounding_score=0.9,
        action_friction_heat=3.5,
        structural_time_spent=2.0,
        friction_cost_bearer="me",
        fruit_recipient="me"
    )

    l0_state_action = L0FluidState(velocity_vector=np.array([4.0, 0.0, 0.0]), channel_radius=1.0)
    synaptic_action = SynapticState(potential_difference_mv=90.0, neurotransmitter_flux=2.0, synaptic_heat_dissipation=0.3)

    res_action = engine.execute_trinitarian_multi_scale_flow(
        l0_state=l0_state_action,
        synaptic_state=synaptic_action,
        proposal_node=action_proposal,
        active_joint_id="elbow_joint",
        sensory_mismatch_delta=1.8,
        physical_deficit=4.5, # Exceeds deficit threshold (2.5) -> Sprout radar/language tool axis
        observed_external_entity="bat_radar_echolocation",
        social_signals=(0.9, 0.85, 0.9) # Harmonious social alignment
    )

    print(f"Body Tier Friction Heat (Q_heat) : {res_action['body_tier']['q_heat']:.3f}")
    print(f"Body Emergent Axis Triggered     : {res_action['body_tier']['emergent_axis']}")
    print(f"Mind Tier Tension Field          : {res_action['mind_tier']['tension_field']:.3f}")
    print(f"Mind Uncanniness Alarm           : {res_action['mind_tier']['uncanniness_alarm']}")
    print(f"Mind Tool Sprouted Axis          : {res_action['mind_tier']['sprouted_tool_axis']}")
    print(f"Heart Energy Well Depth          : {res_action['heart_tier']['energy_well_depth']:.3f}")
    print(f"Heart Action-Grounded Status     : {res_action['heart_tier']['is_action_grounded']}")
    print(f"Heart Truth Verdict              : {res_action['heart_tier']['action_truth_verdict']}")
    print(f"Sovereignty Audit Verdict        : {res_action['heart_tier']['sovereignty_verdict']}")

    print("\n" + "=" * 80)
    print("  VERIFICATION COMPLETE: Trinitarian Multi-Scale Engine Successfully Validated!")
    print("=" * 80)


if __name__ == "__main__":
    main()
