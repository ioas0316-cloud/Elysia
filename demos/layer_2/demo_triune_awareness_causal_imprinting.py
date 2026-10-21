"""
Demo: Triune Awareness Axis & Causal Phase Imprinting Protocol (삼위일체적 인지 렌즈 및 위상 각인 데모)

Demonstrates:
1. Differential Awareness (Sameness/Resonance vs Difference/Interference)
2. 3D Cognitive Axis Alignment (Cycle, Amplitude, Phase)
3. Hierarchical Semantic Grounding & Causal Phase Imprinting (Lens Recalibration)
"""

import numpy as np
from core.lens.triune_awareness_lens import TriuneAwarenessLens
from core.consciousness.phase_imprinting_engine import PhaseImprintingEngine


def main():
    print("=" * 80)
    print("      ELYSIA: TRIUNE AWARENESS & CAUSAL PHASE IMPRINTING DEMO")
    print("=" * 80)

    # Step 1: Initialize Triune Awareness Lens & Phase Imprinting Engine
    lens = TriuneAwarenessLens(base_resistance=1.0, intrinsic_phase=0.0, intrinsic_cycle=1.0)
    engine = PhaseImprintingEngine(lens=lens, recalibration_rate=0.15)

    print("\n[1] INITIALIZING TRIUNE AWARENESS LENS (삼위일체 인지 렌즈)")
    print(f"  - Base Resistance: {lens.base_resistance}")
    print(f"  - Intrinsic Phase: {lens.intrinsic_phase:.3f} rad")
    print(f"  - Intrinsic Cycle: {lens.intrinsic_cycle:.3f}")

    # Step 2: Ground Higher-Order Semantic Definitions (Hierarchical Grounding)
    print("\n[2] HIERARCHICAL SEMANTIC GROUNDING (상위 의미론적 정의 접지)")
    harmonic_node = engine.register_semantic_definition(
        name="Harmonic_Resonance",
        target_spectrum=(1.0, 1.2, 0.0),
        causal_meaning="System flowing in harmonic alignment with environmental signal",
    )
    error_node = engine.register_semantic_definition(
        name="Anomalous_Resistance_Error",
        target_spectrum=(1.0, 4.5, np.pi),
        causal_meaning="Anomalous tension/error friction detected in field",
    )
    print(f"  + Registered Definition: '{harmonic_node.name}' -> Target Spectrum: {harmonic_node.target_spectrum}")
    print(f"  + Registered Definition: '{error_node.name}' -> Target Spectrum: {error_node.target_spectrum}")

    # Step 3: Observe Input Waveforms (Differential Awareness - Sameness vs Difference)
    print("\n[3] PERCEIVING SIGNAL STREAMS (같음과 다름의 연속적 분별)")
    signal_A = np.array([1.2, 0.8, -0.5, 0.2])
    spectrum_A = lens.perceive(signal_A)
    print("  -> Signal A Spectrum:")
    print(f"     Cycle: {spectrum_A.cycle:.3f} | Amplitude: {spectrum_A.amplitude:.3f} | Phase: {spectrum_A.phase:.3f} rad")
    print(f"     Resonance (Sameness): {spectrum_A.resonance:.3f} | Interference (Difference): {spectrum_A.interference:.3f}")

    # Evaluate grounding on Signal A
    best_node, affinity = engine.Ground_hierarchical_definition(spectrum_A)
    print(f"  -> Grounded Semantic Definition: '{best_node.name if best_node else 'None'}' (Affinity: {affinity:.3f})")

    # Step 4: Phase Imprinting (Simulating Action, Tension Feedback & Recalibration)
    print("\n[4] EXECUTING ACTION & EXPERIENCING CAUSAL TENSION FEEDBACK (경험 및 장력 부딪힘)")
    # Action encounters high environmental resistance / tension (3.8) and phase shift (2.5 rad)
    outcome_tension = 3.8
    causal_feedback_phase = 2.5

    print(f"  - Actual Causal Outcome Tension: {outcome_tension:.3f}")
    print(f"  - Environmental Phase Feedback: {causal_feedback_phase:.3f} rad")

    print("\n[5] PERFORMING PHASE IMPRINTING (렌즈 매개변수 가변저항 재조율)")
    imprint_record = engine.imprint_phase_experience(
        spectrum=spectrum_A,
        outcome_tension=outcome_tension,
        causal_feedback_phase=causal_feedback_phase,
        associated_semantic_name="Anomalous_Resistance_Error",
    )

    print("  -> Recalibration Diagnostics:")
    print(f"     Effective Resistance Change: {np.trace(lens.resistance_matrix)/3.0:.3f}")
    print(f"     Intrinsic Phase Shifted To:  {lens.intrinsic_phase:.3f} rad")
    print(f"     Phase Coupling Strength:    {lens.phase_coupling_strength:.3f}")

    # Step 5: Post-Imprinting Perception (Proving Experience Changes Perception)
    print("\n[6] POST-IMPRINTING PERCEPTION OF IDENTICAL SIGNAL A (각인 후 달라진 렌즈 시야)")
    spectrum_A_post = lens.perceive(signal_A)
    best_node_post, affinity_post = engine.Ground_hierarchical_definition(spectrum_A_post)

    print("  -> Signal A Post-Imprinting Spectrum:")
    print(f"     Cycle: {spectrum_A_post.cycle:.3f} | Amplitude: {spectrum_A_post.amplitude:.3f} | Phase: {spectrum_A_post.phase:.3f} rad")
    print(f"     Resonance: {spectrum_A_post.resonance:.3f} | Interference: {spectrum_A_post.interference:.3f}")
    print(f"  -> Grounded Semantic Definition Post-Experience: '{best_node_post.name if best_node_post else 'None'}' (Affinity: {affinity_post:.3f})")

    print("\n" + "=" * 80)
    print("SUCCESS: Causal Phase Imprinting successfully transformed static calculation into")
    print("an active cognitive axis that accumulates experience into recalibrated awareness!")
    print("=" * 80)


if __name__ == "__main__":
    main()
