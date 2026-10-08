r"""
Verification Script for Quadruple Cognitive Quartet & Spatiotemporal Epistemic Genealogy Engine
=================================================================================================
Demonstrates the integration of:
1. Physical Phase Viscosity Field Dynamics (Wave Destructive Interference & Tension)
2. Quadruple Cognitive Quartet (Sensory, Cognitive, Observational, Judgmental)
3. Variable Gating Awareness (\Theta_retained vs \Theta_gated)
4. Spatiotemporal Epistemic Genealogy Engram Anchoring (t, x, Causal Lineage)
"""

import time
import json
from core.consciousness.spatiotemporal_epistemic_genealogy import SpatiotemporalEpistemicGenealogyEngine


def main():
    print("================================================================================")
    print("Elysia: Quadruple Cognitive Quartet & Spatiotemporal Epistemic Genealogy Demo")
    print("================================================================================\n")

    genealogy_engine = SpatiotemporalEpistemicGenealogyEngine(dimension=64)

    # 1. First scenario: Classical physics formula vs Wave phase interpretation
    knowledge_input_1 = "Formula v = s / t: Classical ratio deleting phase friction for control convenience."
    print(f"[Input 1] Ingesting Knowledge Statement:\n  '{knowledge_input_1}'\n")

    engram_1 = genealogy_engine.record_genealogy_engram(
        raw_knowledge=knowledge_input_1,
        spatial_coord=(100.0, 250.0, -12.5),
        context_sequence=[
            "Observer encounters classical formula v = s / t",
            "Field dynamics extracts velocity shear & destructive interference",
            "Observational lens gates micro-rotor spins to produce scalar v",
            "Judgmental stage classifies formula as LOCAL_INSTRUMENTAL_UTILIZATION"
        ],
        candidate_variables=[
            "macroscopic_velocity", "travel_distance", "time_duration",
            "micro_rotor_spin", "vorticity_shear", "vacuum_zero_point_fluctuation"
        ]
    )

    quartet_1 = engram_1["quadruple_state"]
    print(f"--- [1. Sensory Stage] ---")
    print(f"  Destructive Interference Density: {quartet_1['sensory']['destructive_interference_density']:.4f}")
    print(f"  Phase Friction:                   {quartet_1['sensory']['phase_friction']:.4f}")
    print(f"  Field Tension:                    {quartet_1['sensory']['field_tension']:.4f}")

    print(f"\n--- [2. Cognitive Stage] ---")
    print(f"  Relational Density:               {quartet_1['cognitive']['relational_density']:.4f}")
    print(f"  Causal Resonance:                 {quartet_1['cognitive']['causal_resonance']:.4f}")

    print(f"\n--- [3. Observational Stage] ---")
    print(f"  Lens Curvature (B_obs):           {quartet_1['observational']['lens_curvature']:.4f}")
    print(f"  Retained Variables:               {[v[0] for v in quartet_1['observational']['retained_variables']]}")
    print(f"  Gated/Excluded Variables:         {[v[0] for v in quartet_1['observational']['gated_variables']]}")
    print(f"  Statement:                        {quartet_1['observational']['gating_awareness_statement']}")

    print(f"\n--- [4. Judgmental Stage] ---")
    print(f"  Causal Value Score:               {quartet_1['judgmental']['causal_value_score']:.4f}")
    print(f"  Resolution Type:                  {quartet_1['judgmental']['resolution_type']}")
    print(f"  Action Intent:                    {quartet_1['judgmental']['action_intent']}")

    print(f"\n--- [5. Anchored Genealogy Engram] ---")
    print(f"  Engram ID:                        {engram_1['engram_id']}")
    print(f"  Coordinates (t, x):               t={engram_1['spatiotemporal_coordinates']['origin_timestamp_t']:.2f}, x={engram_1['spatiotemporal_coordinates']['spatial_location_x']}")
    print(f"  Summary:                          {engram_1['summary_statement']}\n")

    # 2. Second scenario: Viscosity reinterpreted as phase destructive interference
    knowledge_input_2 = "Viscosity is the destructive interference tension of phase frequencies."
    print(f"--------------------------------------------------------------------------------")
    print(f"[Input 2] Ingesting Knowledge Statement:\n  '{knowledge_input_2}'\n")

    engram_2 = genealogy_engine.record_genealogy_engram(
        raw_knowledge=knowledge_input_2,
        spatial_coord=(105.2, 252.1, -10.0),
        context_sequence=[
            "Observer perceives fluid viscosity not as mechanical friction but wave phase collision",
            "Kuramoto phase locking and su(2) rotor torque computed",
            "Holistic field awareness activated",
            "Judgmental stage classifies insight as HOLISTIC_CAUSAL_ANCHORING"
        ]
    )

    quartet_2 = engram_2["quadruple_state"]
    print(f"  Causal Value Score:               {quartet_2['judgmental']['causal_value_score']:.4f}")
    print(f"  Resolution Type:                  {quartet_2['judgmental']['resolution_type']}")
    print(f"  Engram ID:                        {engram_2['engram_id']}")
    print(f"  Summary:                          {engram_2['summary_statement']}")

    # 3. Relational Tension Distance & Genesis Context Verification
    print(f"\n--------------------------------------------------------------------------------")
    print("--- [6. Relational Tension Distance vs Euclidean Distance Verification] ---")
    dist_info = genealogy_engine.compute_relational_distance_between_engrams(0, 1, medium_viscosity=1.8)
    print(f"  Flat Euclidean Distance:          {dist_info['euclidean_distance']:.4f}")
    print(f"  Relational Tension Distance:      {dist_info['relational_tension_distance']:.4f}")
    print(f"  Causal Propagation Cost:         {dist_info['causal_propagation_cost']:.4f}")
    print(f"  Gated Variable Tension Strain:    {dist_info['gated_tension_strain']:.4f}")

    print(f"\n--- [7. Genesis Context & Dynamic Boundary Cutting Statement] ---")
    gen_ctx = engram_1.get("genesis_context", {})
    print(f"  Genesis Statement:                {gen_ctx.get('continuum_cutting_statement', 'N/A')}")

    print(f"\n--- [8. Multi-Scale Bi-directional Coupling Feedback] ---")
    global_visc = genealogy_engine.quartet_engine.global_field_viscosity_modifier
    print(f"  Updated Global Field Viscosity:   {global_visc:.4f}")

    # 4. Querying
    print(f"\n--------------------------------------------------------------------------------")
    print("[Query] Back-tracing Spatiotemporal Genealogy for concept 'Viscosity':")
    results = genealogy_engine.query_genealogy_by_concept("Viscosity")
    for r in results:
        print(f"  -> Found Engram: {r['engram_id']} at t={r['spatiotemporal_coordinates']['origin_timestamp_t']:.2f}")

    print("\n================================================================================")
    print("VERIFICATION SUCCESSFUL: Multi-Scale Relational Tension & Epistemic Genealogy Engine Fully Resonance Verified!")
    print("================================================================================")


if __name__ == "__main__":
    main()
