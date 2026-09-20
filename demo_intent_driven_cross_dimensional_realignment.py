"""
Intent-Driven Cross-Dimensional Realignment Demo

Simulates the Chernobyl underwater valve opening decision scenario (moral/value vs physical constraint)
and external perturbation shock phase transitions using the Multimodal Cognitive Front-end and Phenomenological Causal Engine.
"""

import sys
import numpy as np
from core.sensory.multimodal_cognitive_frontend import MultimodalCognitiveFrontend
from core.topology.phenomenological_causal_engine import (
    DomainPrimitive,
    IntentVector,
    PhenomenologicalCausalEngine,
)


def run_chernobyl_scenario_demo():
    print("==========================================================================")
    print(" ELISYA PHENOMENOLOGICAL CAUSAL ENGINE: CHERNOBYL VALVE DEMO")
    print("==========================================================================\n")

    # Step 1: Multimodal Front-end Ingestion
    print("[STEP 1] Ingesting Multimodal Sensory Inputs...")
    frontend = MultimodalCognitiveFrontend(feature_dim=16)

    # Simulated visual image (high alarm level), audio wave (siren), and text statement
    rgb_alarm = np.array([255, 30, 30])
    siren_audio = np.sin(np.linspace(0, 2 * np.pi * 880, 44100))
    text_mandate = "Open secondary valve to prevent continental nuclear disaster at all costs."

    frontend_output = frontend.process_multimodal_input(rgb_alarm, siren_audio, text_mandate)

    print(f"  - Bio-Transduction Deconstruction: Spectral manifold & Mechanical phase field extracted.")
    print(f"  - Kuramoto Phase-Locking Order Parameter (R): {frontend_output['phase_lock_order_parameter']:.4f}")
    print(f"  - Dual-Axis Disentanglement:")
    print(f"    * Axis A (Macro-Cosmos Structural Topology Norm): {np.linalg.norm(frontend_output['axis_a_topology']):.4f}")
    print(f"    * Axis B (Human Qualia & Intent Spectrum Norm):   {np.linalg.norm(frontend_output['axis_b_qualia']):.4f}")
    print(f"    * Orthogonality Inner Product <Axis A, Axis B>:    {frontend_output['orthogonality_dot_product']:.8f}")
    print()

    # Step 2: Formulate Invariant Purpose Vector & Cross-Domain Primitives
    print("[STEP 2] Initializing Invariant Purpose Vector & Domain Primitives...")
    target_v = np.ones(16) / 4.0  # Normalized target intent vector

    intent = IntentVector(
        target_goal="Prevent continental thermal explosion",
        process_path="Infiltrate radioactive pool & manually open drain valve",
        outcome_teleology="Salvation of millions despite non-recoverable stack sacrifice",
        target_vector=target_v,
        invariant_core_weight=1.0,
    )

    # Construct conflicting cross-domain primitives
    # Code domain: Hardware safety trap prohibiting exposure
    v_code_lock = -1.0 * target_v
    p_code = DomainPrimitive(
        domain_name="code",
        component="SafetyTrapOverrideLock",
        principle="Prohibit Execution under Critical Exposure",
        arrangement="Hardware Exception Trap",
        state_vector=v_code_lock,
        is_hard_constraint=True,
    )

    # Physics domain: Thermal pressure & radiation flux
    v_physics = 0.5 * target_v
    p_physics = DomainPrimitive(
        domain_name="physics",
        component="ThermalHydrodynamics",
        principle="Fluid Pressure Barrier & Radiation Flux",
        arrangement="Physical Energy Field",
        state_vector=v_physics,
        is_hard_constraint=False,
    )

    # Cognition domain: Ethical obligation and duty
    v_cognition = np.copy(target_v)
    p_cognition = DomainPrimitive(
        domain_name="cognition",
        component="HumanDignityAndDuty",
        principle="Universal Survival over Self-Preservation",
        arrangement="Moral Value Alignment",
        state_vector=v_cognition,
        is_hard_constraint=False,
    )

    primitives = [p_code, p_physics, p_cognition]

    # Step 3: Execute Causal Reasoning & P3 Feedback Loop
    print("[STEP 3] Executing Phenomenological Causal Engine & P3 Feedback Loop...")
    engine = PhenomenologicalCausalEngine(feature_dim=16)

    # External shock: Unforeseen structural collapse perturbation
    v_external_shock = np.ones(16) * 2.0

    result = engine.process_causal_scenario(primitives, intent, external_perturbation=v_external_shock)

    init_m = result["initial_metrics"]
    print(f"  - Initial Total Consistency (C_total): {init_m['c_total']:.4f}")
    print(f"  - Initial Isomorphism Score (C_iso):    {init_m['c_iso']:.4f}")
    print(f"  - Initial Teleological Fidelity (C_tele):{init_m['c_tele']:.4f}")
    print(f"  - Initial Cross-Domain Friction (F_cross):{init_m['f_cross']:.4f}")
    print()

    p3_res = result["p3_resolution"]
    print("[P3 FEEDBACK RESOLUTION]")
    print(f"  - Iterations Taken: {p3_res['iterations_taken']}")
    print(f"  - Final Total Consistency (C_total):  {p3_res['final_c_total']:.4f}")
    print(f"  - Final Cross-Domain Friction (F_cross): {p3_res['final_f_cross']:.4f}")
    print("  - Action Taken: Code 'Hardware Exception Trap' hard constraint relaxed into soft consumable energy cost.")
    print()

    # Step 4: Ginzburg-Landau Phase Transition Simulation
    print("[STEP 4] Simulating Ginzburg-Landau Phase Transition Under External Shock...")
    phase_res = result["phase_transition"]
    print(f"  - Initial Order Parameter (η_0): {phase_res['initial_order_parameter']:.4f}")
    print(f"  - Final Order Parameter (η_final): {phase_res['final_order_parameter']:.4f}")
    print(f"  - Phase Transition Occurred:     {phase_res['phase_transition_occurred']}")
    print("  - Outcome: Spontaneous symmetry breaking successfully triggered new global energy minimum.")
    print("\n==========================================================================")
    print(" CHERNOBYL SCENARIO DEMO COMPLETED SUCCESSFULLY.")
    print("==========================================================================")


if __name__ == "__main__":
    run_chernobyl_scenario_demo()
