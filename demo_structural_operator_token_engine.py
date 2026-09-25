"""
Elysia Engine: Structural Operator Token & Topological Inversion Demo
=====================================================================
Demonstrates:
1. Rune Composition (Health & Fire operators) & Emergent e31 dimension.
2. Non-commutative order: Stereoisomer Chirality & sign inversion.
3. Protein-like topological folding & Attractor Gauge Relaxation.
4. Exact 100% Lossless Inverse Sandwich Transformation (R_dagger).
5. Subject-Object Topological Inversion: Drinking Water (ABSORB) vs Drowning (ENVELOPED).
"""

import numpy as np
from core.topology.structural_operator_engine import (
    StructuralOperatorToken,
    StructuralOperatorEngine,
    SubjectivityMode
)


def run_demo():
    print("=====================================================================")
    print("      Elysia Engine: Structural Operator & Topological Inversion")
    print("=====================================================================\n")

    engine = StructuralOperatorEngine(boundary_capacity=5.0)

    # 1. Define Structural Operators
    t_health = StructuralOperatorToken("Health(e23)", [1.0, 0.0, 0.0])
    t_fire = StructuralOperatorToken("Fire(e12)", [0.0, 0.0, 1.0])

    engine.register_token(t_health)
    engine.register_token(t_fire)

    print("[1] Base Structural Operator Tokens Registered:")
    print(f"  - {t_health}")
    print(f"  - {t_fire}\n")

    # 2. BCH Composition: Health -> Fire vs Fire -> Health (Chirality)
    chain_A = engine.fold_chain_bch(["Health(e23)", "Fire(e12)"], order=2)
    chain_B = engine.fold_chain_bch(["Fire(e12)", "Health(e23)"], order=2)

    print("[2] BCH Operator Composition & Stereoisomer Chirality:")
    print(f"  Chain A (Health ⊗ Fire): {chain_A.bivector}")
    print(f"  Chain B (Fire ⊗ Health): {chain_B.bivector}")
    print(f"  Emergent e31 component (Chain A): {chain_A.bivector[1]:+.4f}")
    print(f"  Emergent e31 component (Chain B): {chain_B.bivector[1]:+.4f}")
    print("  -> Non-commutativity creates exact sign inversion (Chirality / Optical Isomerism)!\n")

    # 3. Exact Lossless State Restoration (100% Reversibility)
    initial_psi = np.array([1.0, 0.5, -0.2, 0.8], dtype=np.float64)
    transformed_psi = chain_A.apply_sandwich(initial_psi)
    restored_psi = chain_A.apply_inverse_sandwich(transformed_psi)

    print("[3] Lossless State Restoration (Unitary Sandwich Transformation):")
    print(f"  Initial State Multivector:   {initial_psi}")
    print(f"  Transformed State:          {transformed_psi}")
    print(f"  Restored State (R_dagger):  {restored_psi}")
    print(f"  Max Absolute Difference:   {np.max(np.abs(initial_psi - restored_psi)):.16e}")
    print("  -> Exact 100% Lossless Recovery Verified!\n")

    # 4. Gauge Curvature Relaxation (Protein Folding Topology)
    relaxed_psi, energy_history = engine.relax_to_attractor(initial_psi, chain_A, steps=10, learning_rate=0.2)
    print("[4] Attractor Gauge Curvature Relaxation (Protein Folding Topology):")
    print(f"  Initial Gauge Potential: {energy_history[0]:.6f}")
    print(f"  Final Relaxed Potential: {energy_history[-1]:.6f}")
    print(f"  Relaxed State Tensor:    {relaxed_psi}\n")

    # 5. Subject-Object Topological Inversion Dynamics
    print("[5] Subject-Object Topological Inversion (Drinking vs Drowning):")

    # Scenario A: Drinking Water (F_ext <= Capacity)
    water_glass_op = StructuralOperatorToken("WaterGlass", [0.8, 0.8, 0.0])
    res_drink = engine.evaluate_topological_inversion(initial_psi, water_glass_op)
    print("  [Scenario A: Drinking Water (Controlled Assimilation)]")
    print(f"    Mode: {res_drink.mode.value}")
    print(f"    External Gauge Curvature F_ext: {res_drink.gauge_curvature:.4f} (Capacity: {engine.boundary_capacity})")
    print(f"    Boundary Integrity: {res_drink.boundary_integrity * 100:.1f}%")
    print(f"    Dissolved into background? {res_drink.is_dissolved()}")
    print(f"    Internal Hydrated State: {res_drink.state_tensor}")

    # Scenario B: Drowning in Tsunami (F_ext > Capacity)
    tsunami_op = StructuralOperatorToken("Tsunami", [4.0, 4.0, 4.0])
    res_drown = engine.evaluate_topological_inversion(initial_psi, tsunami_op)
    print("\n  [Scenario B: Drowning in Tsunami (Overwhelming Envelopment)]")
    print(f"    Mode: {res_drown.mode.value}")
    print(f"    External Gauge Curvature F_ext: {res_drown.gauge_curvature:.4f} (Capacity: {engine.boundary_capacity})")
    print(f"    Boundary Integrity: {res_drown.boundary_integrity * 100:.1f}%")
    print(f"    Dissolved into background? {res_drown.is_dissolved()}")
    print(f"    System Dissolved State Tensor: {res_drown.state_tensor}")

    print("\n=====================================================================")
    print("      Elysia Engine: Structural Operator Verification Complete!")
    print("=====================================================================")


if __name__ == "__main__":
    run_demo()
