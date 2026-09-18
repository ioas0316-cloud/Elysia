"""
Demo script for Triadic Autopoietic Integration Loop.
Demonstrates:
  1. Homeostatic Strain D_H & Thermodynamic Valence V
  2. Active Inference Action Spillover a(t) upon threshold breach
  3. Topological Attractor Memory anchoring & phase locking
  4. Real-time Observability Metrics (R_H, rho_D, dot_S_int, Winding Number)
"""

import numpy as np
from core.consciousness.triadic_autopoietic_engine import (
    TriadicAutopoieticEngine,
    normalize_quaternion,
)


def run_triadic_demo():
    print("===========================================================================")
    print("   ELYSIA: TRIADIC AUTOPOIETIC INTEGRATION LOOP DEMO                       ")
    print("   (Homeostatic Valence + Active Inference + Topological Attractor Memory) ")
    print("===========================================================================\n")

    # Initialize Triadic Autopoietic Engine
    engine = TriadicAutopoieticEngine(
        num_nodes=16,
        action_dim=4,
        homeo_dim=3,
        theta_action=0.25,
        target_homeo=np.array([0.0, 0.0, 0.0]),
        seed=101,
    )

    # Register an Attractor Basin in memory topology
    target_attractor = normalize_quaternion(np.random.randn(16, 4))
    engine.register_attractor(target_attractor)
    print(">>> 0. REGISTERED TOPOLOGICAL ATTRACTOR MEMORY BASIN <<<")
    print(f"  Nodes: {engine.num_nodes}, Action Dim: {engine.action_dim}, Action Threshold: {engine.theta_action}\n")

    # Simulate sequence of boundary sensory inputs q_bound
    # Phase 1: Harmonious boundary input (Low mismatch)
    # Phase 2: High perturbation boundary input (Breaches action threshold)
    # Phase 3: Relaxation and phase-locking
    sensory_inputs = [
        ("Phase 1: Harmonious Input", np.array([1.0, 0.05, 0.05, 0.0])),
        ("Phase 1: Harmonious Input", np.array([1.0, 0.02, 0.02, 0.0])),
        ("Phase 2: Disruptive Perturbation", np.array([0.0, 1.0, 0.5, 0.2])),
        ("Phase 2: Disruptive Perturbation", np.array([0.1, 0.9, 0.8, 0.3])),
        ("Phase 2: Disruptive Perturbation", np.array([0.2, 0.8, 0.7, 0.2])),
        ("Phase 3: Relaxation & Alignment", np.array([0.9, 0.1, 0.1, 0.0])),
        ("Phase 3: Relaxation & Alignment", np.array([0.98, 0.02, 0.02, 0.0])),
        ("Phase 3: Relaxation & Alignment", np.array([1.0, 0.0, 0.0, 0.0])),
    ]

    print(">>> 1. SIMULATING TRIADIC INTEGRATION LOOP OVER TIME <<<")
    for step, (phase_label, q_sensory) in enumerate(sensory_inputs, 1):
        telemetry = engine.step(q_sensory, dt=0.05)
        diag = telemetry["diagnostics"]

        print(f"\n[Step {step}] {phase_label}")
        print(f"  - Phase Tension (T_phase): {telemetry['T_phase']:.4f}")
        print(f"  - Action Spillover (a): {telemetry['action'].round(3)}")
        print(f"  - Homeostatic Strain (D_H): {telemetry['homeostatic_strain']:.4f}")
        print(f"  - Thermodynamic Valence (V): {telemetry['valence']:+.4f}")
        print(f"  - Dynamic Relaxation (gamma): {telemetry['relaxation_gamma']:.3f}")
        print(f"  - Thermal Agitation (T_eff): {telemetry['thermal_T_eff']:.4f}")
        print("  - Diagnostics:")
        print(f"    * Quaternion Order Parameter (R_H): {diag['R_H']:.4f}")
        print(f"    * Gauge Defect Density (rho_D): {diag['rho_D']:.4f}")
        print(f"    * Entropy Dissipation Rate (dot_S_int): {diag['dot_S_int']:.4f}")
        print(f"    * Topological Winding Number: {telemetry['winding_number']:.4f}")

    print("\n===========================================================================")
    print("   TRIADIC AUTOPOIETIC INTEGRATION DEMO COMPLETED SUCCESSFULLY             ")
    print("===========================================================================")


if __name__ == "__main__":
    run_triadic_demo()
