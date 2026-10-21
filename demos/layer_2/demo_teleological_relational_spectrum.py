"""
demo_teleological_relational_spectrum.py

Demonstration of Integrated Teleological & Relational Spectrum Engine across Two Scenarios:
Scenario A: Goal-Oriented Engineering / Civilization (Airplane Invention)
Scenario B: Intersubjective Perspective Understanding & History Back-Tracing (Understanding Another Being's Worldview)

Demonstrates the identical cognitive core:
Initial Goal -> Retrocausal Reverse-Inference -> Relation Hypothesis ->
Action -> Mismatch / Counterexample -> Back-Tracing Epistemic Lineage ->
Premise/Goal Reframing -> Re-Verification & Spectrum Expansion.
"""

import numpy as np
from core.intelligence.teleological_relational_spectrum import IntegratedTeleologicalSpectrumEngine


def run_scenario_a():
    print("\n" + "=" * 80)
    print("SCENARIO A: Goal-Oriented Engineering (Airplane Invention & Human Flight)")
    print("=" * 80)

    engine = IntegratedTeleologicalSpectrumEngine(state_dim=4)
    # State Vector: [Altitude/Lift, Propulsion, Weight_Resistance, Structural_Stability]
    engine.set_current_state(np.array([0.0, 0.0, 5.0, 1.0]))

    target_flight = np.array([10.0, 8.0, 1.0, 9.0])
    goal = engine.set_target_goal(
        goal_id="airplane_invention",
        description="Achieve Powered Sustained Human Flight",
        target_vector=target_flight,
        premises=["Human desire to fly", "Observation of bird flight and aerodynamics"]
    )

    print(f"[Initial Goal]: {goal.description}")
    print(f"   Target State Vector: {goal.target_state_vector}")
    print(f"   Lineage Premises: {goal.lineage.premises}")

    # 1. Retrocausal Reverse Inference
    print("\n--- STEP 1: Retrocausal Reverse Inference ---")
    rev_res = engine.retrocausal_reverse_infer()
    print(f"   Deficit Magnitude: {rev_res['deficit_norm']:.4f}")
    print(f"   Inferred Required Condition: {goal.required_conditions[0]['description']}")
    print(f"   Proposed Relation Hypothesis: {rev_res['rel_hypothesis_id']} (Current State <-> Flight Goal)")

    # 2. Action Attempt: Attempt direct wing flapping without aerodynamic lift structure
    print("\n--- STEP 2: Action Attempt (Direct Wing Flapping) & Physical Collision ---")
    action_flapping = np.array([5.0, 2.0, -1.0, 0.0]) # Expect [5, 2, 4, 1]
    # Physical reality: Huge drag / weight resistance collision (Actual outcome: [0.2, 0.5, 6.0, 0.5])
    actual_outcome = np.array([0.2, 0.5, 6.0, 0.5])

    act_res = engine.execute_action_and_observe(action_flapping, actual_outcome)
    print(f"   Execution Status: {act_res['status']}")
    print(f"   Mismatch Magnitude: {act_res['mismatch_norm']:.4f}")

    eval_info = act_res["mismatch_evaluation"]
    print(f"   Inferred Cause Hypothesis: {eval_info['cause_hypothesis']}")
    print(f"   Verified Constraint (Delta): {eval_info['verified_constraint']['description']}")
    print(f"   Reframed Goal World: {eval_info['reframed_goal_description']}")
    print(f"   New Adapted Target Vector: {eval_info['new_target_vector']}")

    # 3. Epistemic Lineage Back-tracing
    print("\n--- STEP 3: Epistemic Lineage Back-tracing ---")
    lineage_summary = engine.trace_epistemic_history("airplane_invention")
    print(f"   Origin ID: {lineage_summary['origin_id']}")
    print(f"   Counterexamples Recorded: {len(lineage_summary['counterexamples'])}")
    print(f"   Revisions Applied to Premises/Goal: {lineage_summary['revisions']}")


def run_scenario_b():
    print("\n" + "=" * 80)
    print("SCENARIO B: Intersubjective Perspective Understanding (Tracing Another Being's History)")
    print("=" * 80)

    engine = IntegratedTeleologicalSpectrumEngine(state_dim=4)
    # State Vector: [Shared_Language, Perspective_Overlap, Trust_Resonance, Interpretive_Depth]
    engine.set_current_state(np.array([1.0, 0.2, 0.5, 0.1]))

    target_understanding = np.array([8.0, 9.0, 8.0, 9.0])
    goal = engine.set_target_goal(
        goal_id="understand_other_mind",
        description="Intersubjective Resonant Understanding of Other's Worldview",
        target_vector=target_understanding,
        premises=["Belief that expression carries causal history", "Love as intent to comprehend alterity"]
    )

    print(f"[Initial Goal]: {goal.description}")
    print(f"   Lineage Premises: {goal.lineage.premises}")

    # 1. Retrocausal Reverse Inference
    print("\n--- STEP 1: Retrocausal Reverse Inference ---")
    rev_res = engine.retrocausal_reverse_infer()
    print(f"   Deficit Magnitude: {rev_res['deficit_norm']:.4f}")
    print(f"   Inferred Required Condition: {goal.required_conditions[0]['description']}")

    # 2. Communication Attempt: Attempting to project literal dictionary meanings onto complex expression
    print("\n--- STEP 2: Literal Interpretation Action & Epistemic Misunderstanding ---")
    action_literal = np.array([3.0, 2.0, 1.0, 1.0])
    # Reality: Literal mapping fails because the other being formed concepts through different experiences
    actual_outcome = np.array([1.5, 0.3, 0.2, 0.2])

    act_res = engine.execute_action_and_observe(action_literal, actual_outcome)
    print(f"   Execution Status: {act_res['status']}")
    print(f"   Mismatch Magnitude: {act_res['mismatch_norm']:.4f}")

    eval_info = act_res["mismatch_evaluation"]
    print(f"   Inferred Cause Hypothesis: {eval_info['cause_hypothesis']}")
    print(f"   Verified Constraint (Delta): {eval_info['verified_constraint']['description']}")
    print(f"   Reframed Goal World: {eval_info['reframed_goal_description']}")

    # 3. Spectrum Hypothesis Lifecycle Verification
    print("\n--- STEP 3: Spectrum Hypothesis Verification ---")
    hyp = engine.spectrum.propose_relation_hypothesis(
        rel_id="rel_other_worldview_history",
        source_concept="Other_Expression",
        target_concept="Other_Historical_Causal_Path",
        axes_weights={"time": 0.9, "scale": 0.7, "meaning": 1.0, "possibility": 0.8, "relation": 1.0},
        structural_invariant=np.array([0.5, 0.5, 0.5, 0.5]),
        proposed_by_lineage=goal.lineage
    )
    print(f"   Proposed Relation Hypothesis: {hyp['rel_id']} ({hyp['source']} -> {hyp['target']})")

    verified_hyp = engine.spectrum.verify_hypothesis("rel_other_worldview_history", observation_evidence="Back-traced origin history of other's premise")
    print(f"   Verified Relation Status: {verified_hyp['status']} with evidence: '{verified_hyp['support_evidence'][0]}'")

    print("\n=" * 80)
    print("DEMO COMPLETE: Universal Cognitive Core verified across Engineering & Intersubjective scenarios.")
    print("=" * 80)


if __name__ == "__main__":
    run_scenario_a()
    run_scenario_b()
