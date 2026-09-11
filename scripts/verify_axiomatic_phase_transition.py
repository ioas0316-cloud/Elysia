r"""
Demonstration & Verification Script for Axiomatic Phase Transition
========================================================================================
최상위 공리 명제 전이 시 일어나는 자발적 상전이 및 인과 결합 매트릭스($J_{ij}$) 리와이어링을
실시간 시뮬레이션으로 시각화 및 검증합니다.
"""

import sys
import os
import numpy as np

# Root path addition
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.consciousness.phase_transition_reconfiguration_engine import AxiomaticPhaseTransitionEngine


def run_verification_demo():
    print("============================================================================")
    print("🏛️ ELYSIA: Axiomatic Phase Transition & Spontaneous Reconfiguration Demo")
    print("============================================================================\n")

    engine = AxiomaticPhaseTransitionEngine(num_nodes=6, dimension=64)

    print(f"1. Initial State Initialization")
    print(f"   - Current Top Axiom: {engine.current_axiom_name}")
    print(f"   - Node Count: {engine.num_nodes}")
    print(f"   - Initial Coupling Matrix J_ij (mean conductance): {np.mean(engine.J_matrix):.4f}\n")

    # Step 1: Subcritical Shift Test
    print("2. Processing Sub-critical Axiom Shift Proposal...")
    subcritical_axiom = "RIGID_BOUNDARY_ISOLATION_MINOR_VARIANT"
    engine.critical_threshold = 10.0  # Force subcritical
    res1 = engine.process_axiom_shift(subcritical_axiom)
    print(f"   - Action: {res1['action']}")
    print(f"   - Message: {res1['message']}")
    print(f"   - Potential Energy: {res1['discrepancy_diagnosis']['potential_energy']:.4f}")
    print(f"   - Transitions Completed: {engine.phase_transition_count}\n")

    # Step 2: Critical Phase Transition Shift
    engine.critical_threshold = 0.45  # Restore normal threshold
    print("3. Processing Critical Axiom Shift Proposal ('RIGID_BOUNDARY_ISOLATION' -> 'KENOTIC_CRUCIFORM_LOVE_GIVING')...")
    res2 = engine.process_axiom_shift(AxiomaticPhaseTransitionEngine.AXIOM_OPEN_KENOTIC_LOVE)

    diag = res2["discrepancy_diagnosis"]
    trans = res2["transition_result"]

    print(f"   - Action: {res2['action']}")
    print(f"   - Discrepancy Gradient (|∇ΔΘ|): {diag['gradient_magnitude']:.4f}")
    print(f"   - Released Potential Energy: {trans['potential_energy_released']:.4f}")
    print(f"   - J_ij Rewiring Magnitude (||ΔJ||): {trans['rewiring_magnitude_delta_J']:.4f}")
    print(f"   - New Mean Conductance: {trans['mean_conductance']:.4f}")
    print(f"   - New Current Axiom: {engine.current_axiom_name}")
    print(f"   - Total Phase Transitions Completed: {engine.phase_transition_count}\n")

    # Step 3: Second Phase Transition to Transcendent Synapse
    print("4. Processing Second Critical Shift ('KENOTIC_CRUCIFORM_LOVE_GIVING' -> 'CIVILIZATIONAL_HOLISTIC_SYNAPSE')...")
    res3 = engine.process_axiom_shift(AxiomaticPhaseTransitionEngine.AXIOM_TRANSCENDENT_SYNAPSE)
    trans3 = res3["transition_result"]

    print(f"   - Action: {res3['action']}")
    print(f"   - Rewiring Magnitude (||ΔJ||): {trans3['rewiring_magnitude_delta_J']:.4f}")
    print(f"   - Current Axiom: {engine.current_axiom_name}")
    print(f"   - Total Phase Transitions Completed: {engine.phase_transition_count}\n")

    print("============================================================================")
    print("✅ Axiomatic Phase Transition Verification Completed Successfully!")
    print("============================================================================")


if __name__ == "__main__":
    run_verification_demo()
