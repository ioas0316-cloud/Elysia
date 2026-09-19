"""
Demonstration script for Observational Substrate & Phenomenology Mirror.

Exemplifies non-reductionist contemplation and Causal Symbol Deconstruction:
1. Contemplating micro-entities ('Amount', 'Flow', 'Resistance') as independent causal structures with Telos & Directionality.
2. Observing boundary traversal friction & interaction.
3. Deconstructing completed external symbols ('Flow Rate', 'Viscosity') into lower-order micro-components and procedural coupling principles.
4. Establishing Causal Necessity and internalizing true Knowledge Causality.
5. Mapping emergent higher-order causal structures ('Flow Rate') without forcing scalar multiplication formulas.
6. Discernment of Homological Stem ('같음의 줄기') vs Disparate Branches ('다름의 가지') between external reality and internal code/formulas.
"""

from core.lens.observational_substrate import ObservationalSubstrate


def run_phenomenology_demonstration():
    print("==========================================================================")
    print(" Elysia Observational Substrate: Causal Deconstruction & Mirroring Demo")
    print("==========================================================================")

    mirror = ObservationalSubstrate("Elysia_Phenomenology_Mirror")

    # 1. Contemplate independent micro-entities
    print("\n--- [Step 1: Contemplating External Entities (Telos & Directionality)] ---")
    e_amount = mirror.contemplate_entity(
        name="Amount",
        telos_purpose="Spatial presence and substantive density",
        directionality="Stationary aggregation within local state boundary",
        density_amount=10.0,
        boundary_tenacity=2.5,
        mobility_vector=[0.0, 0.0, 0.0],
        chromatic_signature=[0.1, 0.8, 0.1]
    )
    print(f"  * Entity 'Amount': Telos = '{e_amount.telos_purpose}' | Direction = '{e_amount.directionality}'")

    e_flow = mirror.contemplate_entity(
        name="Flow",
        telos_purpose="Boundary traversal & temporal progression",
        directionality="Directional transit across spatial boundaries over time",
        density_amount=3.0,
        boundary_tenacity=1.0,
        mobility_vector=[2.0, 0.5, 0.0],
        chromatic_signature=[0.8, 0.1, 0.1]
    )
    print(f"  * Entity 'Flow': Telos = '{e_flow.telos_purpose}' | Direction = '{e_flow.directionality}'")

    # 2. Deconstruct External Symbol & Internalize Knowledge Causality
    print("\n--- [Step 2: Causal Symbol Deconstruction (역해체) & Internalization] ---")
    deconstructed = mirror.deconstruct_external_symbol(
        external_symbol="Flow Rate",
        micro_component_names=["Amount", "Flow"],
        procedural_coupling_principle="Spatial Amount traversing boundary under Flow motion",
        causal_necessity_statement="Flow Rate exists necessarily when spatial Amount is driven across boundaries by Flow"
    )
    print(f"  * Deconstructed External Symbol: '{deconstructed.external_symbol}'")
    print(f"    - Micro-Components: {[c.name for c in deconstructed.micro_components]}")
    print(f"    - Procedural Coupling: {deconstructed.procedural_coupling_principle}")
    print(f"    - Causal Necessity Statement: {deconstructed.causal_necessity_statement}")

    knowledge = mirror.internalize_knowledge_causality("Flow Rate")
    print(f"  * Internalized Knowledge Causality:")
    print(f"    - Invariant Lineage: {knowledge.internal_invariant_lineage}")
    print(f"    - Necessity Proof: {knowledge.causal_necessity_proof}")

    # 3. Observe boundary coupling & friction
    print("\n--- [Step 3: Observing Boundary Traversal & Friction] ---")
    trace = mirror.observe_coupling("Amount", "Flow", boundary_medium_tenacity=0.8)
    print(f"  * Coupling Trace (Amount <-> Flow):")
    print(f"    - Friction Resistance: {trace.friction_resistance:.4f}")
    print(f"    - Boundary Deformation: {trace.boundary_deformation:.4f}")
    print(f"    - Energy Dissipation: {trace.energy_dissipation:.4f}")

    # 4. Map emergent structure ('Flow Rate')
    print("\n--- [Step 4: Mapping Emergent Causal Structure ('Flow Rate')] ---")
    emergent = mirror.map_emergent_structure(
        emergent_name="Flow Rate",
        originating_names=["Amount", "Flow"],
        emergent_telos="Quantitative structural transit per boundary crossing",
        directional_extension="Continuous spatiotemporal stream spanning state boundaries",
        boundary_traces=[trace]
    )
    print(f"  * Emergent Structure: {emergent.emergent_name}")
    print(f"    - Emergent Telos: {emergent.emergent_telos}")
    print(f"    - Directional Extension: {emergent.directional_extension}")
    print(f"    - Invariant Backbone: {emergent.invariant_backbone}")

    # 5. Homological discernment (Stem & Branch)
    print("\n--- [Step 5: Homological Discernment (Stem & Branch Dissection)] ---")
    internal_procedural_frame = {
        "backbone": "flow_rate = velocity * area",
        "context_branches": ["Procedural_Script", "Static_Rulebase_Formula"]
    }

    discernment = mirror.discern_homology("Flow Rate", internal_procedural_frame)
    print(f"  * Homological Stem ('같음의 줄기'):")
    print(f"    -> {discernment.homological_stem}")
    print(f"  * Disparate Branches ('다름의 가지'):")
    for branch in discernment.disparate_branches:
        print(f"    -> {branch}")
    print(f"  * Reason for Divergence:")
    print(f"    -> {discernment.reason_for_difference}")
    print(f"  * Isomorphism Fidelity: {discernment.isomorphism_fidelity:.4f}")

    # 6. Generate Contemplative Reflection
    print("\n--- [Step 6: Contemplative Reflection] ---")
    reflection = mirror.generate_contemplative_reflection("Flow Rate")
    print(reflection)


if __name__ == "__main__":
    run_phenomenology_demonstration()
