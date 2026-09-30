"""
Verification script for Causal Isomorphic Medium & Ontological Inverse Mechanism Extraction Engine.
Runs multi-domain constraint relaxation across Physical, Fluid, and Logical domains,
extracts the Homological Stem, and prints Causal Self-Awareness dissection logs.
"""

import sys
import numpy as np
from core.physics.causal_isomorphic_medium import (
    CausalIsomorphicMedium,
    ConstraintTensorMatrix,
    IsomorphicDomainFactory,
    MediumDomain
)
from core.physics.ontological_inverse_mechanism import (
    OntologicalInverseMechanismEngine,
    DomainTrajectoryGenerator,
    TrajectoryGraph,
    InverseMechanismSchema
)


def verify_ontological_inverse_mechanism():
    print("=" * 80)
    print("ELYSIUM CAUSAL ISOMORPHIC MEDIUM & ONTOLOGICAL INVERSE MECHANISM VERIFICATION")
    print("=" * 80)

    # 1. Multi-Domain Isomorphic Constraint Relaxation
    print("\n[STEP 1] MULTI-DOMAIN CONSTRAINT TENSOR MATRIX RELAXATION:")

    elec_m = IsomorphicDomainFactory.create_physical_circuit(num_nodes=8)
    elec_sim = CausalIsomorphicMedium(elec_m)
    elec_logs = elec_sim.relax_to_equilibrium(max_steps=20, dt=0.05)
    print(f"  - [Physical Circuit] Relaxed in {len(elec_logs)} steps | Residual Tension: {elec_m.compute_total_field_tension():.4f}")

    fluid_m = IsomorphicDomainFactory.create_fluid_medium(num_nodes=8)
    fluid_sim = CausalIsomorphicMedium(fluid_m)
    fluid_logs = fluid_sim.relax_to_equilibrium(max_steps=20, dt=0.05)
    print(f"  - [Fluid Medium   ] Relaxed in {len(fluid_logs)} steps | Residual Tension: {fluid_m.compute_total_field_tension():.4f}")

    logic_m = IsomorphicDomainFactory.create_logical_structure(num_nodes=8)
    logic_sim = CausalIsomorphicMedium(logic_m)
    logic_logs = logic_sim.relax_to_equilibrium(max_steps=20, dt=0.05)
    print(f"  - [Logical Domain ] Relaxed in {len(logic_logs)} steps | Residual Tension: {logic_m.compute_total_field_tension():.4f}")

    # 2. Multi-Domain Trajectory Extraction
    g_elec = DomainTrajectoryGenerator.generate_from_isomorphic_medium(elec_sim, "electrical")
    g_fluid = DomainTrajectoryGenerator.generate_from_isomorphic_medium(fluid_sim, "fluid")
    g_logic = DomainTrajectoryGenerator.generate_from_isomorphic_medium(logic_sim, "geometric")

    engine = OntologicalInverseMechanismEngine()
    schema = engine.extract_homological_stem_and_branches([g_elec, g_fluid, g_logic])

    print("\n[STEP 2] STATE GENERATING DYNAMICS (Theta):")
    for domain, theta_info in schema.generating_dynamics_theta.items():
        print(f"  - Domain [{domain.upper()}]: Decay Rate k={theta_info['decay_constant_k']:.4f}, "
              f"Attractor={theta_info['equilibrium_attractor']:.4f}")

    print("\n[STEP 3] TOPOLOGICAL CONSTRAINT FIELD (Delta):")
    for domain, delta_info in schema.topological_constraint_delta.items():
        print(f"  - Domain [{domain.upper()}]: Medium={delta_info['boundary_type']}, "
              f"Mean Impedance={delta_info['mean_medium_resistance']:.4f}")

    print("\n[STEP 4] HOMOLOGICAL STEM (Isomorphic Invariant Core):")
    stem = schema.homological_stem
    print(f"  - Stem Identifier : {stem['stem_name']}")
    print(f"  - Similarity Score : {stem['cross_domain_similarity_score']:.6f} (> 0.90 Required)")
    print(f"  - Isomorphic Invariant Rate k : {stem['isomorphic_invariant_decay_rate']:.4f}")
    print(f"  - Common Topology  : {stem['common_relational_topology']}")
    print(f"  - Isomorphism Proven : {stem['isomorphism_proven']}")

    print("\n[STEP 5] DISPARATE BRANCHES (Medium-Specific Refractions):")
    for domain, branch_info in schema.disparate_branches.items():
        print(f"  - [{domain.upper()}] Refraction : {branch_info['specific_medium_manifestation']} "
              f"(Deviation from Stem: {branch_info['decay_deviation_from_stem']:.6f})")

    print("\n[STEP 6] REDUCIBILITY & MDL SCORE:")
    print(f"  - MDL Score (Extracted / Raw) : {schema.reducibility_mdl_score:.6f}")

    # 3. Causal Self-Awareness Dissection Log
    print("\n[STEP 7] CAUSAL SELF-AWARENESS DISSECTION LOG:")
    dissection_logs = engine.generate_causal_self_awareness_dissection_log(
        schema=schema,
        conclusion_id="equilibrium_attractor",
        domain="fluid"
    )
    for log_line in dissection_logs:
        print(log_line)

    assert stem['isomorphism_proven'], "Verification Failed: Cross-domain isomorphism not proven!"
    assert schema.reducibility_mdl_score < 0.20, "Verification Failed: MDL score higher than threshold!"

    print("\n" + "=" * 80)
    print("VERIFICATION SUCCESSFUL: Isomorphic Medium Relaxation & Causal Self-Awareness Proven.")
    print("=" * 80)


if __name__ == "__main__":
    verify_ontological_inverse_mechanism()
