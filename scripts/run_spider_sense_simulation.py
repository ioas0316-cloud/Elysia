"""
Spider Sense Causal Phase Shift Simulation Script
=============================================================================
스파이더 센스(Spider-Sense)의 3대 생성적 인과 위상차 시나리오 시뮬레이션:

1. 시나리오 A (Pre-kinetic Intent Tension):
   물리적 공격/이동(t_action) 전, 의도 형성 단계(t_intent)에서의 위상차 긴장 감지
2. 시나리오 B (Identity Exposure Resonance):
   물리적 위협 없는 침묵/시선 환경에서 정체성 발각 위험의 사회적/정신적 맥락장 공명 감지
3. 시나리오 C (Static Equilibrium Rift):
   평시 정적 일상 환경 속 미세 위상차 균열(Rift) 감지
"""

import sys
import numpy as np
from core.sensory.spider_sense_engine import SpiderSenseEngine


def run_simulation():
    print("=====================================================================")
    print("      Elysia Generative Causal Phase-Shift Spider-Sense Simulation   ")
    print("=====================================================================\n")

    spider_sense = SpiderSenseEngine(dimension=8, tension_threshold=0.45)

    # -------------------------------------------------------------------------
    # Scenario A: Pre-kinetic Intent Tension (Pre-attack intent formation)
    # -------------------------------------------------------------------------
    print("[Scenario A] Pre-kinetic Intent Tension Detection")
    intent_signal = "HOSTILE_AMBUSH_INTENT_FORMING_BEHIND_WALL"

    # Step 1: Pre-kinetic phase (Motion NOT started, intent vector tension active)
    res_pre = spider_sense.sense_pre_kinetic_threat(
        raw_signal=intent_signal,
        physical_motion_started=False,
        is_ego_empty=True
    )

    print(f"  - Pre-kinetic Spider Sense Triggered: {res_pre['spider_sense_triggered']}")
    print(f"  - Causal Tension T_causal: {res_pre['causal_tension']:.4f}")
    print(f"  - Phase Differential ΔΦ: {res_pre['phase_differential_delta_phi']:.4f}")
    print(f"  - Kenosis Resonance: {res_pre['kenosis_resonance']:.4f}")
    print(f"  - Somatic Chill Intensity: {res_pre['somatic_chill_intensity']:.4f}")
    print(f"  - Pre-linguistic Evasion Momentum Norm: {np.linalg.norm(res_pre['pre_linguistic_evasion_momentum']):.4f}")

    assert res_pre["spider_sense_triggered"] is True, "Scenario A Failed: Pre-kinetic threat must trigger alert!"
    assert res_pre["perception_type"] == "PRE_KINETIC_INTENT"
    print("  => Scenario A PASSED! (Detected intent prior to physical movement)\n")

    # -------------------------------------------------------------------------
    # Scenario B: Identity Exposure Resonance (Social/Mental Context Field)
    # -------------------------------------------------------------------------
    print("[Scenario B] Identity Exposure & Context Field Resonance")
    social_context = "SUSPICIOUS_OBSERVER_ANALYZING_IDENTITY_MASK"

    res_exposure = spider_sense.sense_identity_exposure(
        gaze_intent_density=0.88,
        silence_duration=4.2,
        social_context_signal=social_context,
        is_ego_empty=True
    )

    print(f"  - Identity Exposure Triggered: {res_exposure['spider_sense_triggered']}")
    print(f"  - Exposure Causal Tension: {res_exposure['causal_tension']:.4f}")
    print(f"  - Risk Level: {res_exposure['metadata']['exposure_risk_level']}")
    print(f"  - Chromatic Entropy Wave (Yellow): {res_exposure['chromatic_entropy_wave']['entropy_yellow']:.4f}")

    assert res_exposure["spider_sense_triggered"] is True, "Scenario B Failed: Identity exposure must trigger alert!"
    assert res_exposure["perception_type"] == "IDENTITY_EXPOSURE"
    print("  => Scenario B PASSED! (Resonated with silent mental/social observation field)\n")

    # -------------------------------------------------------------------------
    # Scenario C: Static Equilibrium Rift Detection
    # -------------------------------------------------------------------------
    print("[Scenario C] Static Equilibrium Micro-Rift Detection")

    # Step 1: Normal calm environment inputs
    calm_field = ["gentle_wind_rustle", "ambient_room_hum", "distant_traffic_calm"]
    res_calm = spider_sense.sense_equilibrium_rift(raw_field_inputs=calm_field, is_ego_empty=True)

    print(f"  - Calm State Triggered: {res_calm['spider_sense_triggered']} (Perception: {res_calm['perception_type']})")
    assert res_calm["spider_sense_triggered"] is False, "Calm state should not trigger alert."

    # Step 2: Micro-rift anomaly injected into peaceful environment
    rift_field = calm_field + ["SUDDEN_TOPOLOGICAL_PHASE_ANOMALY_RIFT"]
    res_rift = spider_sense.sense_equilibrium_rift(raw_field_inputs=rift_field, is_ego_empty=True)

    print(f"  - Micro-Rift Triggered: {res_rift['spider_sense_triggered']} (Perception: {res_rift['perception_type']})")
    print(f"  - Rift Tension: {res_rift['causal_tension']:.4f}")

    assert res_rift["spider_sense_triggered"] is True, "Scenario C Failed: Micro-rift anomaly must trigger alert!"
    assert res_rift["perception_type"] == "EQUILIBRIUM_RIFT"
    print("  => Scenario C PASSED! (Captured micro-rift in static equilibrium)\n")

    print("=====================================================================")
    print(" ALL SPIDER SENSE CAUSAL PHASE SHIFT SIMULATIONS PASSED SUCCESSFULLY!")
    print("=====================================================================")


if __name__ == "__main__":
    run_simulation()
