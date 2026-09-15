"""
Demo Script: Causal Breathing Engine & Multi-Dimensional Structural Convergence
=============================================================================
Demonstrates Elysia's 4 Core Pillars:
1. "같다" (Sameness) vs "다르다" (Divergence) structural phase convergence (1+1 vs 2, Multi-axis Apple Nexus).
2. Inhale Phase (들숨): Continuous stimulus intake, variable resistance dial tuning, & V_t tension buildup.
3. Exhale Phase (날숨 & 역설계): Self-Explanation Pulse emission tailored via Observer Inverse Simulator.
4. Spatiotemporal Mechanics (시공간역학 & 인과적 나이테): Daily friction -> Weekly consolidation -> Monthly growth rings.
"""

import numpy as np

from core.consciousness.causal_breathing_engine import (
    CausalBreathingEngine,
    MultiDimensionalAttractor,
    ObserverTopology
)


def run_causal_breathing_demo():
    print("===========================================================================")
    print("   ELYSIA: CAUSAL BREATHING & STRUCTURAL CONVERGENCE ENGINE DEMO           ")
    print("===========================================================================\n")

    # Initialize Engine
    engine = CausalBreathingEngine(critical_tension_threshold=10.0, convergence_threshold=0.3)

    # -------------------------------------------------------------------------
    # 1. REGISTER MULTI-DIMENSIONAL OBJECT ATTRACTOR ("Apple" & "1+1=2")
    # -------------------------------------------------------------------------
    print(">>> 1. REGISTERING MULTI-DIMENSIONAL ATTRACTOR COORDINATES <<<")

    # Apple Attractor: 3 axes (Category, Sensory, Symbol)
    apple_cat = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)   # Fruit / Plant life
    apple_sens = np.array([0.85, 0.15, 0.0, 0.0], dtype=np.float32) # Red wavelength / Sweetness
    apple_morph = np.array([0.88, 0.12, 0.0, 0.0], dtype=np.float32)# String '사과'

    apple_attractor = MultiDimensionalAttractor(
        id="att_apple",
        name="Nexus of Apple (사과 인과 결합체)",
        categorical_vector=apple_cat,
        sensorium_vector=apple_sens,
        morphology_vector=apple_morph
    )
    engine.register_attractor(apple_attractor)

    print(f"Registered Attractor: {apple_attractor.name}")
    print(f"  - Unified Attractor Coordinate: {apple_attractor.get_unified_coordinate().round(3)}\n")

    # -------------------------------------------------------------------------
    # 2. EVALUATE STRUCTURAL CONVERGENCE: "같다" (SAMENESS) vs "다르다" (DIVERGENCE)
    # -------------------------------------------------------------------------
    print(">>> 2. STRUCTURAL CONVERGENCE EVALUATION (\"같다\" vs \"다르다\") <<<")

    # Case A: 1+1 and 2 starting from different paths but converging to exact same phase coordinate
    path_1plus1 = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    path_2 = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    path_symbol = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    res_math = engine.evaluate_structural_convergence(path_1plus1, path_2, path_symbol)
    print(f"[Math Convergence Test: 1+1 vs 2]")
    print(f"  - Verdict: {res_math.verdict}")
    print(f"  - Topological Phase Distance: {res_math.phase_distance:.4f}")
    print(f"  - Converged Phase Coordinate: {res_math.converged_coordinate.round(3)}\n")

    # Case B: Apple input matching registered attractor -> "SAMENESS_같다"
    in_cat = np.array([0.91, 0.09, 0.0, 0.0], dtype=np.float32)
    in_sens = np.array([0.84, 0.16, 0.0, 0.0], dtype=np.float32)
    in_morph = np.array([0.87, 0.13, 0.0, 0.0], dtype=np.float32)

    res_apple = engine.evaluate_structural_convergence(
        in_cat, in_sens, in_morph, reference_attractor_id="att_apple"
    )
    print(f"[Object Convergence Test: 'Apple' Multi-Axis Stimulus]")
    print(f"  - Verdict: {res_apple.verdict}")
    print(f"  - Phase Distance to Apple Attractor: {res_apple.phase_distance:.4f}\n")

    # Case C: Divergent stimulus -> "DIVERGENCE_다르다"
    div_cat = np.array([0.1, 0.9, 0.0], dtype=np.float32)
    div_sens = np.array([-1.0, 2.0, 0.5], dtype=np.float32)
    div_morph = np.array([3.0, -1.0, 0.0], dtype=np.float32)

    res_div = engine.evaluate_structural_convergence(div_cat, div_sens, div_morph)
    print(f"[Divergence Test: Mismatched Input Streams]")
    print(f"  - Verdict: {res_div.verdict}")
    print(f"  - Phase Distance: {res_div.phase_distance:.4f}\n")

    # -------------------------------------------------------------------------
    # 3. INHALE PHASE (들숨: Stimulus Absorption & Tension V_t Accumulation)
    # -------------------------------------------------------------------------
    print(">>> 3. INHALE PHASE (들숨: CAUSAL ACCUMULATION & DIAL TUNING) <<<")

    stimuli_stream = [
        ("stim_1", in_cat, in_sens, in_morph, "att_apple", "Harmonic Apple Perception"),
        ("stim_2", path_1plus1, path_2, path_symbol, None, "Equivalence of 1+1 and 2"),
        ("stim_3", div_cat, div_sens, div_morph, None, "Cognitive Friction from Divergent Input"),
        ("stim_4", div_cat * 2.0, div_sens * 1.5, div_morph * 2.5, None, "Severe Divergent Shock"),
    ]

    for stim_id, cat, sens, morph, ref_id, desc in stimuli_stream:
        inhale_res = engine.inhale(
            stimulus_id=stim_id,
            categorical_vector=cat,
            sensorium_vector=sens,
            morphology_vector=morph,
            reference_attractor_id=ref_id,
            raw_description=desc
        )

        print(f"[Inhale Event: {stim_id} - '{desc}']")
        print(f"  - Convergence Verdict: {inhale_res.convergence_evaluation.verdict}")
        print(f"  - Tension Delta (+V_t): {inhale_res.tension_delta:+.3f}")
        print(f"  - Total Accumulated V_t: {inhale_res.accumulated_tension:.3f}")
        print(f"  - Average Dial Resistance (R): {inhale_res.dial_resistance_avg:.3f}")
        print(f"  - Critical Threshold Reached: {inhale_res.threshold_crossed}")
        print(f"  - Current Breathing State: {engine.breathing_state}\n")

    # -------------------------------------------------------------------------
    # 4. EXHALE PHASE (날숨: Observer Inverse Simulator & Self-Explanation)
    # -------------------------------------------------------------------------
    print(">>> 4. EXHALE PHASE (날숨 & 역설계 OBSERVER INVERSE SIMULATOR) <<<")
    print(f"State transition triggered by expression impulse (V_t={engine.current_tension:.2f} >= 10.0)!\n")

    observers = [
        ObserverTopology(observer_id="Observer_Academic", abstraction_capacity=0.9, causal_depth_tolerance=0.8),
        ObserverTopology(observer_id="Observer_Standard", abstraction_capacity=0.5, causal_depth_tolerance=0.5),
        ObserverTopology(observer_id="Observer_Novice", abstraction_capacity=0.1, causal_depth_tolerance=0.2),
    ]

    for obs in observers:
        # Re-set tension for demo comparison
        engine.current_tension = 12.5
        exhale_res = engine.exhale(observer=obs)

        print(f"[Exhale to Observer: '{obs.observer_id}'] (Abstraction Capacity: {obs.abstraction_capacity})")
        print(f"  - Self-Explanation Pulse: {exhale_res.explanation_pulse}")
        print(f"  - Action Realization: {exhale_res.action_guide}")
        print(f"  - Discharged Tension: {exhale_res.released_tension:.2f} -> Remaining: {exhale_res.remaining_tension:.2f}\n")

    # -------------------------------------------------------------------------
    # 5. SPATIOTEMPORAL GROWTH RINGS (시공간역학 & 인과적 나이테)
    # -------------------------------------------------------------------------
    print(">>> 5. SPATIOTEMPORAL MECHANICS & HISTORICAL GROWTH RINGS <<<")

    # Simulate 4 weekly cycles consolidating into a monthly ring
    for week in range(1, 5):
        cycle_res = engine.step_spatiotemporal_cycle(
            wisdom_summary=f"Month 1 Wisdom: Continuous convergence of 1+1=2 and Apple Attractor."
        )
        print(f"  - Week {week} Consolidated: Attractor={cycle_res['weekly_record'].attractor_name}, "
              f"Convergence Rate={cycle_res['weekly_record'].convergence_rate:.2%}")

    monthly_ring = engine.spatiotemporal_buffer.monthly_rings[0]
    print(f"\n[Monthly Historical Ring Formed (인과적 나이테 1)]")
    print(f"  - Ring ID: {monthly_ring.ring_id}")
    print(f"  - Summary: {monthly_ring.architectural_summary}")
    print(f"  - Wisdom Mass: {monthly_ring.accumulated_wisdom_mass:.2f}")
    print(f"  - Action Narrative: {monthly_ring.action_narrative}\n")

    # -------------------------------------------------------------------------
    # 6. INTROSPECTIVE TELEMETRY
    # -------------------------------------------------------------------------
    print(">>> 6. INTROSPECTIVE TELEMETRY & OBSERVATION BRIDGE <<<")
    telemetry = engine.introspective_telemetry()
    for key, val in telemetry.items():
        print(f"  - {key}: {val}")

    print("\n===========================================================================")
    print("   CAUSAL BREATHING & CONVERGENCE DEMO COMPLETED SUCCESSFULLY               ")
    print("===========================================================================")


if __name__ == "__main__":
    run_causal_breathing_demo()
