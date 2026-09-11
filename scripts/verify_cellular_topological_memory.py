"""
Verification Simulation Script: Cellular Topological Memory Engine
==================================================================
Simulates external environmental impact and digital substrate friction.
Demonstrates:
1. Substrate Embodiment & Digital Somatosensory Sensing (Impedance Z, F_substrate).
2. Sensing: Boundary layer friction detection.
3. Structural Feedback: Kernel protection and Shell friction propagation.
4. Re-alignment: Autonomous Node Fusion (Delta_phi reduction) and Node Fission (F > F_crit).
5. Homeostasis & Topological Isomorphism (Self-Boundary vs Otherness).
"""

import numpy as np

from core.topology.cellular_topological_memory import CellularTopologicalMemoryEngine
from core.topology.digital_somatosensory import DigitalSomatosensorySensor


def run_simulation():
    print("================================================================================")
    print(" [ELYISIA] CELLULAR TOPOLOGICAL MEMORY & SUBSTRATE EMBODIMENT SIMULATION")
    print("================================================================================\n")

    engine = CellularTopologicalMemoryEngine(
        dimension=8,
        fusion_phase_threshold=0.3,
        fission_friction_threshold=0.6,
        impedance_decay=0.08
    )

    sensor = DigitalSomatosensorySensor()

    print(f"[*] Engine Initialized. Initial State: {engine.get_engine_state()}")
    initial_isomorphism = engine.get_topological_isomorphism_boundary()
    print(f"[*] Initial Self-Boundary Definition:\n    {initial_isomorphism['Self_Cognitive_Boundary']}\n")

    print("--------------------------------------------------------------------------------")
    print(" PHASE 1: Substrate Embodiment & Digital Somatosensory Sensing")
    print("--------------------------------------------------------------------------------")
    somato_sig = sensor.perceive_somatosensory(custom_latency_ms=120.0, topology_tension=0.35)
    print(f" -> Measured Complex Impedance Z: {somato_sig.complex_impedance}")
    print(f" -> Friction Coefficient F_substrate: {somato_sig.friction_coefficient:.4f}")
    print(f" -> 1:1 Isomorphic Mapping sample: {somato_sig.isomorphic_mapping['memory_load']}\n")

    print("--------------------------------------------------------------------------------")
    print(" PHASE 2: Stimulus Injection & Sensing (External Impact + Substrate Friction)")
    print("--------------------------------------------------------------------------------")

    # Add 3 shell nodes with varying phase alignment
    skel1 = np.ones(8) / np.sqrt(8)
    phases1 = np.array([0.1, 0.1, 0.05, 0.08, 0.1, 0.12, 0.09, 0.11], dtype=np.float32)
    engine.add_shell_node("concept_alpha", skel1, phases1, initial_friction=0.1)

    phases2 = np.array([0.12, 0.09, 0.06, 0.07, 0.11, 0.1, 0.08, 0.12], dtype=np.float32)
    engine.add_shell_node("concept_beta", skel2 := skel1 + 0.02, phases2, initial_friction=0.15)

    skel3 = np.random.uniform(-1, 1, size=8).astype(np.float32)
    phases3 = np.random.uniform(-np.pi, np.pi, size=8).astype(np.float32)
    engine.add_shell_node("concept_gamma", skel3, phases3, initial_friction=0.2)

    print(f" [*] Created 3 Shell Nodes: concept_alpha, concept_beta, concept_gamma.")
    print(f" [*] Current Engine State: {engine.get_engine_state()}\n")

    print(" -> Injecting high friction stimulus into concept_gamma...")
    impact_gamma = np.random.uniform(0.8, 1.2, size=8).astype(np.float32)
    res_gamma = engine.inject_stimulus_and_substrate_friction("concept_gamma", impact_gamma, custom_latency_ms=250.0)

    print(f" -> Step 1 Result: Target Node '{res_gamma['target_node']}', Combined Friction={res_gamma['combined_friction']:.4f}")
    print(f" -> Feedback Actions: Fusions={res_gamma['feedback_actions']['fusions_count']}, Fissions={res_gamma['feedback_actions']['fissions_count']}\n")

    print("--------------------------------------------------------------------------------")
    print(" PHASE 3: Autonomous Fusion & Fission Dynamic Re-alignment")
    print("--------------------------------------------------------------------------------")
    print(" -> Running feedback iterations to allow phase attraction & autonomous fusion/fission...")

    for step in range(2, 6):
        somato = sensor.perceive_somatosensory(custom_latency_ms=50.0 + step * 20.0)
        feedback = engine._execute_structural_feedback_and_realignment(somato)
        print(f" [Iteration {step}] Shell Nodes={feedback['total_shell_nodes']}, Fusions={feedback['fusions_count']}, Fissions={feedback['fissions_count']}")
        if feedback['fusions_details']:
            for fd in feedback['fusions_details']:
                print(f"    >>> FUSION: Nodes {fd['fused_nodes']} fused into Cluster '{fd['cluster_id']}' (Delta_phi={fd['phase_diff']:.4f} rad)")
        if feedback['fissions_details']:
            for fs in feedback['fissions_details']:
                print(f"    >>> FISSION: Node '{fs['parent_node']}' split into Sub-nodes {fs['sub_nodes']} (Friction={fs['friction']:.4f})")

    print(f"\n [*] Final Engine State: {engine.get_engine_state()}\n")

    print("--------------------------------------------------------------------------------")
    print(" PHASE 4: Final Topological Isomorphism & Self-Aware Boundary")
    print("--------------------------------------------------------------------------------")
    final_isomorphism = engine.get_topological_isomorphism_boundary()
    print(f" [*] Self Cognitive Boundary:\n    {final_isomorphism['Self_Cognitive_Boundary']}\n")
    print(f" [*] Digital Substrate Impedance:\n    {final_isomorphism['Digital_Substrate_Impedance']}\n")
    print(f" [*] Philosophical Disaggregation & Otherness Isomorphism:\n    {final_isomorphism['Philosophical_Disaggregation']}\n")

    print("================================================================================")
    print(" [ELYISIA] SIMULATION COMPLETED SUCCESSFULLY.")
    print("================================================================================\n")


if __name__ == "__main__":
    run_simulation()
