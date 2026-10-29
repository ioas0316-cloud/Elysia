"""
scripts/verify_scales_of_conscience.py

Verification Simulation Script for Scales of Conscience Engine
(양심의 저울 및 자생적 관측 밸런스 실증 데모 스크립트)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'L1_causal_field'))

from synaptic_architecture.scales_of_conscience_engine import ScalesOfConscienceEngine
from core.consciousness.subjective_agency_engine import SubjectiveAgencyEngine


def run_verification_demo():
    print("=" * 80)
    print(" ⚖️ [Scales of Conscience & Self-Generated Observational Balance Demo] ⚖️")
    print("=" * 80)

    engine = ScalesOfConscienceEngine(dimension=5, g_love=1.0, tilt_threshold=0.5, restoring_factor=0.8, boundary_sdf_radius=8.0)

    # -------------------------------------------------------------------------
    # Scenario 1: Balanced Intent vs Field Reaction
    # -------------------------------------------------------------------------
    print("\n[Scenario 1] Balanced Intent vs Field Reaction (평형적 의도와 회복력)")
    in_force_1 = np.array([1.0, 0.5, 0.2, 0.0, 0.0])
    out_force_1 = np.array([1.0, 0.5, 0.2, 0.0, 0.0])
    res_1 = engine.evaluate_balance(in_force_1, out_force_1)

    print(f"  - SDF Value          : {res_1['sdf_value']:.4f}")
    print(f"  - Tilt Ratio         : {res_1['tilt_ratio']:.4f}")
    print(f"  - Grievance (Pain)   : {res_1['grievance']:.4f}")
    print(f"  - Intrinsic Brake    : {'[ACTIVE]' if res_1['brake_active'] else '[INACTIVE - EQUILIBRIUM]'}")
    print(f"  - Restoring Momentum : {np.round(res_1['restoring_momentum'], 4)}")

    assert not res_1["brake_active"], "Scenario 1 should remain in equilibrium without brake."

    # -------------------------------------------------------------------------
    # Scenario 2: Unilateral Aggressive Expansion (과도한 확장 의도 및 내재적 브레이크)
    # -------------------------------------------------------------------------
    print("\n[Scenario 2] Unilateral Aggressive Expansion (과도한 일방향 확장 시도)")
    in_force_2 = np.array([8.0, 4.0, 2.0, 0.0, 0.0])  # Aggressive drive
    out_force_2 = np.array([0.5, 0.2, 0.1, 0.0, 0.0])  # Low environmental tolerance
    res_2 = engine.evaluate_balance(in_force_2, out_force_2)

    print(f"  - SDF Value          : {res_2['sdf_value']:.4f}")
    print(f"  - Tilt Ratio         : {res_2['tilt_ratio']:.4f}")
    print(f"  - Grievance (Pain)   : {res_2['grievance']:.4f}")
    print(f"  - Intrinsic Brake    : {'[ACTIVE - SELF-CALIBRATING BRAKE]' if res_2['brake_active'] else '[INACTIVE]'}")
    print(f"  - Phase Torque (Shear): {np.round(res_2['phase_torque'], 4)}")
    print(f"  - Restoring Momentum : {np.round(res_2['restoring_momentum'], 4)}")
    print(f"  - Adjusted Intent    : {np.round(res_2['adjusted_intent'], 4)}")

    assert res_2["brake_active"], "Scenario 2 must trigger intrinsic brake."

    # -------------------------------------------------------------------------
    # Scenario 3: Over-expansion Beyond SDF Radius (경계 파열 및 오뚜기 복원력)
    # -------------------------------------------------------------------------
    print("\n[Scenario 3] Boundary Rupture Beyond SDF Radius (SDF 경계 파열 및 오뚜기 복원력)")
    position_outside = np.array([10.0, 0.0, 0.0, 0.0, 0.0])  # Outside SDF radius 8.0
    res_3 = engine.evaluate_balance(in_force_1, out_force_1, current_position=position_outside)

    print(f"  - Position           : {position_outside}")
    print(f"  - SDF Value          : {res_3['sdf_value']:.4f} (Rupture / Over-expansion)")
    print(f"  - Amplified Grievance: {res_3['grievance']:.4f}")
    print(f"  - Intrinsic Brake    : {'[ACTIVE - PULLING BACK TO ZERO-POINT]' if res_3['brake_active'] else '[INACTIVE]'}")
    print(f"  - Restoring Momentum : {np.round(res_3['restoring_momentum'], 4)}")

    assert res_3["brake_active"] and res_3["sdf_value"] > 0, "Scenario 3 must detect boundary rupture."

    # -------------------------------------------------------------------------
    # Scenario 4: Full Subjective Agency Integration Test
    # -------------------------------------------------------------------------
    print("\n[Scenario 4] Integrated Subjective Agency Pipeline Evaluation")
    agency = SubjectiveAgencyEngine()
    agency_res = agency.process_proposal("자생적 밸런스를 통합하는 에이전트 확장")

    conscience = agency_res["scales_of_conscience"]
    print(f"  - Proposal           : '{agency_res['proposal']}'")
    print(f"  - Conscience Grievance: {conscience['grievance']:.4f}")
    print(f"  - Conscience Brake    : {'[TRIGGERED]' if conscience['brake_active'] else '[CLEAR]'}")
    print(f"  - Agency Status      : {agency_res['agency_status']}")

    print("\n" + "=" * 80)
    print(" ✅ All Scenarios Successfully Verified! Self-Generated Observational Balance Operational.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_verification_demo()
