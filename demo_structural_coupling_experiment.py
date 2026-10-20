"""
Interactive Demo Script for ELYSIA Structural Coupling & Discrepancy Tracking v0.2
Demonstrates the 4-Step Structural Coupling & Proof Protocol Evolution:
1. Initial Relation Inference (1 -> 2 -> 3) and Proof Protocol Logging
2. Observation & Discrepancy Analysis (Noise vs Structural Discrepancy vs Context Conflict)
3. Execution of the 4 Structural Reactions: MAINTAIN, MODIFY, BRANCH (Mitosis), SUSPEND_JUDGMENT
4. Closed-loop Re-observation & Future Prediction Evolution
"""

from core.topology.structural_coupling_engine import StructuralCouplingEngine


def run_demo():
    print("=" * 75)
    print("ELYSIA Structural Coupling & Proof Protocol Engine v0.2 Demo")
    print("=" * 75)

    engine = StructuralCouplingEngine(noise_threshold=0.15)

    # Step 1: Record initial relation 1 -> 2 -> 3
    print("\n--- STEP 1: Initial Relation Inference & Proof Protocol ---")
    initial_seq = ["1", "2", "3"]
    engine.record_initial_relation(initial_seq)
    print(f"Recorded initial relational sequence: {' -> '.join(initial_seq)}")

    pred_1 = engine.primary_graph.predict_next_state("1")
    pred_2 = engine.primary_graph.predict_next_state("2")
    print(f"Initial Prediction from '1': {pred_1}")
    print(f"Initial Prediction from '2': {pred_2}")

    print("\n[Proof Protocol Log (Step 1)]")
    for step in engine.primary_graph.proof_protocol.steps:
        print(f"  Step {step.step_index}: [{step.rule_applied}] {step.justification_notes}")

    # Step 2 & 3: Reaction 1 - MAINTAIN on Noise
    print("\n--- STEP 2 & 3 (Case A): Observation with Minor Noise (delta = 0.05) ---")
    res_a = engine.process_observation(
        current_state="1",
        observed_next_state="2",
        observed_value_delta=0.05
    )
    print(f"Discrepancy Type: {res_a['analysis'].discrepancy_type}")
    print(f"Structural Reaction: {res_a['reaction']}")
    print(f"Action Details: {res_a['details']['action']}")

    # Step 2 & 3: Reaction 2 - SUSPEND_JUDGMENT on Ambiguous Data
    print("\n--- STEP 2 & 3 (Case B): Ambiguous / Insufficient Observation Data ---")
    res_b = engine.process_observation(
        current_state="2",
        observed_next_state="UNKNOWN_X",
        is_ambiguous_data=True
    )
    print(f"Discrepancy Type: {res_b['analysis'].discrepancy_type}")
    print(f"Structural Reaction: {res_b['reaction']}")
    print(f"Action Details: {res_b['details']['action']}")

    # Step 2 & 3: Reaction 3 - BRANCH (Mitosis) on Mutually Exclusive Context
    print("\n--- STEP 2 & 3 (Case C): Context Conflict (2 -> 5 under 'high_gravity') ---")
    res_c = engine.process_observation(
        current_state="2",
        observed_next_state="5",
        context_condition="high_gravity"
    )
    print(f"Discrepancy Type: {res_c['analysis'].discrepancy_type}")
    print(f"Structural Reaction: {res_c['reaction']}")
    print(f"Action Details: {res_c['details']['action']}")

    # Step 2 & 3: Reaction 4 - MODIFY on Structural Discrepancy (2 -> 4)
    print("\n--- STEP 2 & 3 (Case D): Uncontextualized Structural Discrepancy (2 -> 4) ---")
    res_d = engine.process_observation(
        current_state="2",
        observed_next_state="4",
        observed_value_delta=1.0
    )
    print(f"Discrepancy Type: {res_d['analysis'].discrepancy_type}")
    print(f"Structural Reaction: {res_d['reaction']}")
    print(f"Action Details: {res_d['details']['action']}")

    # Step 4: Re-observation & Future Prediction Evolution
    print("\n--- STEP 4: Re-observation & Prediction Protocol Evolution ---")
    new_pred_main = engine.predict_future_under_layer("2", "main_layer")
    new_pred_branch = engine.predict_future_under_layer("2", "context_high_gravity")

    print(f"Evolved Prediction from '2' under 'main_layer': {new_pred_main} (Changed from '3' -> '4')")
    print(f"Evolved Prediction from '2' under 'context_high_gravity': {new_pred_branch}")

    print("\n[Proof Protocol Log (Complete History)]")
    for step in engine.primary_graph.proof_protocol.steps:
        print(f"  Step {step.step_index}: [{step.rule_applied}] {step.justification_notes}")

    print("\n" + "=" * 75)
    print("SUCCESS: 4-Step Structural Coupling & Proof Protocol Experiment Completed!")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()
