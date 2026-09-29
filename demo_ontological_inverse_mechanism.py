"""
Interactive CLI Demo for Ontological Inverse Mechanism Extraction.
Demonstrates reverse-engineering multi-domain result trajectories G = (V, E, C)
and extracting the Homological Stem across Fluid, Electrical, and Geometric fields.
"""

import time
from core.physics.ontological_inverse_mechanism import OntologicalInverseMechanismEngine


def main():
    print("=" * 80)
    print(" ELYSIA ONTOLOGICAL INVERSE MECHANISM EXTRACTION DEMO")
    print(" 'Do not calculate, observe results and extract generating dynamics.'")
    print("=" * 80)

    time.sleep(0.3)
    print("\n>>> Step 1: Observing Raw Causal Trajectories G = (V, E, C) from 3 Domains...")
    print("  1) Fluid Medium: Wave displacement, impedance interface refraction, viscosity friction")
    print("  2) Electrical Circuit: Voltage potential drop, potentiometer dial resistance gradient")
    print("  3) Geometric Space: Fermat/Pythagorean lattice topological strain relaxation")

    engine = OntologicalInverseMechanismEngine()

    time.sleep(0.3)
    print("\n>>> Step 2: Executing Inverse Mechanism Extraction (Theta, Delta, Stem, Branches)...")
    schema = engine.run_multi_domain_inverse_extraction()

    time.sleep(0.3)
    print("\n>>> Step 3: Extracted 4-Layer Principle Invariants:")
    print("-" * 60)
    print(" [Layer 1 - State Generating Dynamics (Theta)]")
    for domain, theta in schema.generating_dynamics_theta.items():
        print(f"   * Domain {domain:10s} | Equation: {theta['governing_equation']}")
        print(f"                            | Decay k: {theta['decay_constant_k']:.4f}, Attractor: {theta['equilibrium_attractor']:.4f}")

    print("\n [Layer 2 - Topological Constraint Field (Delta)]")
    for domain, delta in schema.topological_constraint_delta.items():
        print(f"   * Domain {domain:10s} | Boundary: {delta['boundary_type']:25s} | Mean Resistance: {delta['mean_medium_resistance']:.4f}")

    print("\n [Layer 3 - Homological Stem (같음의 줄기 - Isomorphic Invariant Core)]")
    stem = schema.homological_stem
    print(f"   * Stem Name          : {stem['stem_name']}")
    print(f"   * Cross Similarity   : {stem['cross_domain_similarity_score']:.6f} / 1.000000")
    print(f"   * Invariant Decay k  : {stem['isomorphic_invariant_decay_rate']:.4f}")
    print(f"   * Isomorphism Proven : {stem['isomorphism_proven']}")
    print(f"   * Common Topology    : {stem['common_relational_topology']}")

    print("\n [Layer 4 - Disparate Branches (다름의 가지 - Medium Refractions)]")
    for domain, branch in schema.disparate_branches.items():
        print(f"   * [{domain.upper():10s}] -> {branch['specific_medium_manifestation']} (Refraction deviation: {branch['decay_deviation_from_stem']:.6f})")

    print("\n [MDL & Reducibility Score]")
    print(f"   * Score: {schema.reducibility_mdl_score:.6f} (Minimal representation extracted)")

    print("\n" + "=" * 80)
    print(" DEMO COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
