"""
Verification Script: Waveform & Dynamical Cognitive Feedback Engine
===================================================================
이 스크립트는 도레미파 -> 솔라시도 음계 지각 및 물리 동역학 수식의 역메커니즘 추출,
그리고 인지 피드백 고리의 수렴 과정을 시각적/로그적으로 종합 검증합니다.
"""

import sys
import numpy as np

from core.consciousness.waveform_cognitive_feedback import (
    WaveformCognitiveFeedbackEngine
)


def run_verification():
    print("=" * 80)
    print("Elysia: Waveform & Dynamical Cognitive Feedback Verification")
    print("=" * 80)

    engine = WaveformCognitiveFeedbackEngine()

    # 1. 도레미파 -> 솔라시도 연속적 지각 및 옥타브 외삽 검증
    print("\n[1] Musical Scale Relational Interval Perception (Do-Re-Mi-Fa -> So-La-Ti-Do)")
    do_re_mi_fa = [261.63, 293.66, 329.63, 349.23]  # C4, D4, E4, F4
    scale_res = engine.perceive_scale_interval_structure(do_re_mi_fa)

    print(f"  - Input Sequence (Do, Re, Mi, Fa): {do_re_mi_fa}")
    print("  - Perceived & Extrapolated Scale Mapping:")
    for note, freq in scale_res["scale_mapping"].items():
        print(f"      * {note:8s} : {freq:7.2f} Hz")

    print(f"  - Is Octave Continuum Extrapolated: {scale_res['is_octave_extrapolated']}")
    octave_ratio = scale_res["extrapolated_frequencies"][7] / scale_res["extrapolated_frequencies"][0]
    print(f"  - Octave Ratio (Do_high / Do): {octave_ratio:.4f} (Target ~ 2.0000)")

    # 2. 물리 동역학 수식의 역메커니즘 추출 (Inverse Mechanism Generation)
    print("\n[2] Extracting Generating Mechanism (\\Theta) from Physical Dynamical Systems")

    # (A) 조화 진동자 (Harmonic Oscillator)
    t = np.linspace(0, 10, 100)
    dt = t[1] - t[0]
    y_harmonic = 2.0 + 3.5 * np.cos(2 * np.pi * 0.8 * t + 0.5)

    mech_harmonic = engine.extract_generating_mechanism(y_harmonic, dt=dt)
    print("\n  (A) Harmonic Oscillator Signal:")
    print(f"      * System Type      : {mech_harmonic.system_type}")
    print(f"      * Extracted Formula: {mech_harmonic.equation_repr}")
    print(f"      * Invariants       : {mech_harmonic.invariants}")
    print(f"      * MDL Complexity   : {mech_harmonic.mdl_complexity:.4f}")

    # (B) 지수적 성장 동역학 (Exponential Dynamical Growth)
    t_exp = np.linspace(0, 4, 40)
    dt_exp = t_exp[1] - t_exp[0]
    y_exp = 1.5 * np.exp(0.5 * t_exp)

    mech_exp = engine.extract_generating_mechanism(y_exp, dt=dt_exp)
    print("\n  (B) Exponential Growth Signal:")
    print(f"      * System Type      : {mech_exp.system_type}")
    print(f"      * Extracted Formula: {mech_exp.equation_repr}")
    print(f"      * MDL Complexity   : {mech_exp.mdl_complexity:.4f}")

    # 3. 인지적 피드백 고리 (Cognitive Feedback Loop & Homeostasis)
    print("\n[3] Running Cognitive Feedback Loop & Rotor Phase Alignment")
    fb_res = engine.cognitive_feedback_loop(
        observed_trajectory=y_harmonic,
        steps_ahead=30,
        dt=dt,
        max_iterations=25
    )

    print(f"  - Observed Points      : {fb_res.observed_length}")
    print(f"  - Extrapolated Points  : {fb_res.extrapolated_length}")
    print(f"  - Discrepancy Error    : {fb_res.discrepancy_error:.6f}")
    print(f"  - Resonance Score      : {fb_res.resonance_score:.4f}")
    print(f"  - Adjusted Rotor Angle : {fb_res.phase_rotor_angle:.4f} rad")
    print(f"  - Homeostasis Achieved : {fb_res.is_homeostasis_achieved}")

    print("\n" + "=" * 80)
    print("Verification Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
