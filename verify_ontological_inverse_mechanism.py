"""
Verification script for Ontological Inverse Mechanism Extraction Engine.
Runs multi-domain trajectory reverse-engineering across Fluid, Electrical,
and Geometric constraint fields to verify Homological Stem extraction.
"""

import sys
from core.physics.ontological_inverse_mechanism import OntologicalInverseMechanismEngine


def verify_ontological_inverse_mechanism():
    print("=" * 80)
    print("ELYSIUM ONTOLOGICAL INVERSE MECHANISM EXTRACTION VERIFICATION")
    print("=" * 80)

    engine = OntologicalInverseMechanismEngine()
    schema = engine.run_multi_domain_inverse_extraction()

    print("\n[1] STATE GENERATING DYNAMICS (Theta):")
    for domain, theta_info in schema.generating_dynamics_theta.items():
        print(f"  - Domain [{domain.upper()}]: Decay Rate k={theta_info['decay_constant_k']:.4f}, "
              f"Attractor={theta_info['equilibrium_attractor']:.4f}")

    print("\n[2] TOPOLOGICAL CONSTRAINT FIELD (Delta):")
    for domain, delta_info in schema.topological_constraint_delta.items():
        print(f"  - Domain [{domain.upper()}]: Medium={delta_info['boundary_type']}, "
              f"Mean Resistance={delta_info['mean_medium_resistance']:.4f}")

    print("\n[3] HOMOLOGICAL STEM (Isomorphic Invariant Core):")
    stem = schema.homological_stem
    print(f"  - Stem Identifier : {stem['stem_name']}")
    print(f"  - Similarity Score : {stem['cross_domain_similarity_score']:.6f} (> 0.90 Required)")
    print(f"  - Isomorphic Invariant Rate k : {stem['isomorphic_invariant_decay_rate']:.4f}")
    print(f"  - Common Topology  : {stem['common_relational_topology']}")
    print(f"  - Isomorphism Proven : {stem['isomorphism_proven']}")

    print("\n[4] DISPARATE BRANCHES (Medium-Specific Refractions):")
    for domain, branch_info in schema.disparate_branches.items():
        print(f"  - [{domain.upper()}] Refraction : {branch_info['specific_medium_manifestation']} "
              f"(Deviation from Stem: {branch_info['decay_deviation_from_stem']:.6f})")

    print("\n[5] REDUCIBILITY & MDL SCORE:")
    print(f"  - MDL Score (Extracted / Raw) : {schema.reducibility_mdl_score:.6f}")

    assert stem['isomorphism_proven'], "Verification Failed: Cross-domain isomorphism not proven!"
    assert schema.reducibility_mdl_score < 0.1, "Verification Failed: MDL score higher than threshold!"

    print("\n" + "=" * 80)
    print("VERIFICATION SUCCESSFUL: Homological Stem & Disparate Branches extracted.")
    print("=" * 80)


if __name__ == "__main__":
    verify_ontological_inverse_mechanism()
