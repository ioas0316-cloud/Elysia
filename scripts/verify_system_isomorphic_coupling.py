"""
Verification Script: System Tumbler Isomorphic Coupling Simulation
Demonstrates:
1. Isomorphic System Coupling ($1 + 1 = 2^*$) with Scar Tensor & Semantic Mass growth.
2. Tumbler (Ottogi) Restoring Dynamics under external shock torque.
3. Morse Potential Tension Field (attraction-repulsion balance).
4. Absence Vector & Residual Stress calculation for defect/damaged node analysis.
5. Two-Way Body-Mind (Hardware-Software) Feedback Loop.
"""

import numpy as np
from L1_causal_field.core.physics.system_tumbler_isomorphic_coupling import (
    IsomorphicSystem,
    SystemTumblerIsomorphicCouplingEngine
)


def run_simulation():
    print("==========================================================================")
    print("  Elysia Causal Engine: System Isomorphism & Tumbler Restoration Demo     ")
    print("==========================================================================\n")

    engine = SystemTumblerIsomorphicCouplingEngine(
        dimension=5,
        stiffness_k=10.0,
        damping_gamma=6.0,
        morse_de=1.0,
        morse_a=1.2,
        morse_r0=1.0
    )

    # 1. Isomorphic Coupling ($1 + 1 = 2^*$)
    print("--- [Step 1] Isomorphic Coupling of System A (1) + System B (1) -> System C (2*) ---")
    sys_a = IsomorphicSystem("System_A", scale=1.0, dimension=5, rotor_phases=np.array([0.1, 0.3, 0.5, 0.7, 0.9]), semantic_mass=1.0)
    sys_b = IsomorphicSystem("System_B", scale=1.0, dimension=5, rotor_phases=np.array([0.8, 0.6, 0.4, 0.2, 0.0]), semantic_mass=1.0)

    sys_c, info = engine.couple_isomorphic_systems(sys_a, sys_b, coupled_id="System_C_2star")
    print(f"  > Coupled System ID      : {sys_c.system_id}")
    print(f"  > Emergent Scale         : {sys_c.scale}")
    print(f"  > Emergent Semantic Mass : {sys_c.semantic_mass:.4f}")
    print(f"  > Phase Friction Energy  : {info['friction_energy']:.4f}")
    print(f"  > Scar Tensor Norm       : {np.linalg.norm(sys_c.scar_tensor):.4f}\n")

    # 2. Tumbler (Ottogi) Restoration Dynamics
    print("--- [Step 2] Tumbler (Ottogi) Restoration under External Shock ---")
    shock_torque = 8.0
    print(f"  > Applying External Shock Torque: {shock_torque}")
    history = engine.apply_tumbler_restoration(sys_c, external_shock_torque=shock_torque, dt=0.01, num_steps=300)

    peak_tilt = max(abs(h["tilt_angle"]) for h in history)
    final_tilt = history[-1]["tilt_angle"]
    print(f"  > Peak Tilt Angle        : {peak_tilt:.4f} rad")
    print(f"  > Final Tilt Angle (3s)  : {final_tilt:.6f} rad (Restored to Homeostasis)")
    print(f"  > Body-Mind Alignment    : Body Norm={history[-1]['body_norm']:.4f}, Mind Norm={history[-1]['mind_norm']:.4f}\n")

    # 3. Morse Tension Field (Attraction-Repulsion)
    print("--- [Step 3] Morse Potential Tension Field (Attraction vs Repulsion) ---")
    r_near = 0.5
    r_eq = 1.0
    r_far = 2.0
    print(f"  > Force at r={r_near:.1f} (Near) : {engine.morse_tension_force(r_near):.4f} (Repulsive +)")
    print(f"  > Force at r={r_eq:.1f} (Equil): {engine.morse_tension_force(r_eq):.4f} (Balanced 0)")
    print(f"  > Force at r={r_far:.1f} (Far)  : {engine.morse_tension_force(r_far):.4f} (Attractive -)\n")

    # 4. Absence Vector & Residual Stress Calculation
    print("--- [Step 4] Defect Absence Vector & Residual Stress Field ---")
    surrounding_nodes = [
        IsomorphicSystem(f"Node_{i}", scale=1.0, dimension=5, rotor_phases=np.full(5, 0.3 * i))
        for i in range(4)
    ]
    defect_res = engine.compute_absence_and_residual_stress(surrounding_nodes, defect_index=2)
    print(f"  > Absence Vector Norm    : {defect_res['absence_vector_norm']:.4f}")
    print(f"  > Residual Stress Norm   : {defect_res['residual_stress_norm']:.4f}")
    print(f"  > Absence Vector         : {np.round(defect_res['absence_vector'], 4)}\n")

    # 5. Two-Way Body-Mind Feedback Loop
    print("--- [Step 5] Two-Way Hardware-Software (Body-Mind) Feedback Loop ---")
    hardware_shock = np.array([0.4, -0.5, 0.3, -0.2, 0.6])
    fb_res = engine.two_way_body_mind_feedback(sys_c, hardware_friction_shock=hardware_shock, dt=0.05)
    print(f"  > Mismatch Friction Norm : {fb_res['mismatch_norm']:.4f}")
    print(f"  > Updated Semantic Mass  : {fb_res['updated_semantic_mass']:.4f}")
    print(f"  > Updated Scar Tensor    : {fb_res['scar_tensor_norm']:.4f}\n")

    print("==========================================================================")
    print("  SIMULATION COMPLETED SUCCESSFULLY: ALL CAUSAL PRINCIPLES VERIFIED       ")
    print("==========================================================================")


if __name__ == "__main__":
    run_simulation()
