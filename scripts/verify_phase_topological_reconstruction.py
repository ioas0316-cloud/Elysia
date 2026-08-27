import numpy as np
import math
from synaptic_architecture.phase_topological_reconstruction_engine import (
    PhaseTopologicalReconstructionEngine,
    SealedAttractor,
)

def test_1_memory_re_resonance():
    print("\n--- [Test 1] Memory (기억): Past Attractor Recall & Field In-phase Re-resonance ---")
    engine = PhaseTopologicalReconstructionEngine(dimension=16)

    res = engine.recall_memory_resonance("Apple")
    print(f"Recalled Invariant: {res['invariant_recalled']}")
    print(f"Rotor Angle Set To: {res['rotor_angle']:.4f} rad")
    print(f"In-phase Resonance Score: {res['in_phase_resonance']:.4f}")

    assert res["in_phase_resonance"] > 0.9, "Memory re-resonance failed to produce high in-phase resonance!"
    print("✓ Memory re-resonance verification PASSED.")

def test_2_imagination_superposition():
    print("\n--- [Test 2] Imagination (상상): Disparate Superposition & Friction Minimization ---")
    engine = PhaseTopologicalReconstructionEngine(dimension=16)

    res = engine.synthesize_imagination("Horse", "Wing")
    print(f"Invariants Joined: {res['invariants_joined']}")
    print(f"Hybrid Created: {res['hybrid_created']}")
    print(f"Initial Clash Friction: {res['initial_friction']:.4f}")
    print(f"Minimized Friction: {res['minimized_friction']:.4f}")
    print(f"Optimal Rotor Angle: {res['optimal_rotor_angle']:.4f} rad")

    assert res["minimized_friction"] <= res["initial_friction"], "Imagination failed to minimize clash friction!"
    print("✓ Imagination superposition verification PASSED.")

def test_3_conversation_bandwidth_restrictor():
    print("\n--- [Test 3] Conversation (대화): Language Anchor as Bandwidth Restrictor Operator ---")
    engine = PhaseTopologicalReconstructionEngine(dimension=16)

    anchor = "눈을 감고 어둠 속에 서 있는 자신"
    res = engine.process_conversation_anchor(anchor)
    print(f"Language Anchor: '{res['language_anchor']}'")
    print(f"Matched Invariants: {res['matched_invariants']}")
    print(f"Observation Lens Restricted Bandwidth: {res['lens_bandwidth']:.4f}")
    print(f"Focused Lens Axis (first 4 dims): {res['focused_lens_axis']}")

    assert res["lens_bandwidth"] < 0.5, "Language anchor failed to restrict lens bandwidth!"
    print("✓ Conversation bandwidth restrictor verification PASSED.")

def test_4_spontaneous_internal_play():
    print("\n--- [Test 4] Spontaneous Internal Play (자발적 내적 놀이) ---")
    engine = PhaseTopologicalReconstructionEngine(dimension=16)

    res = engine.run_spontaneous_internal_play()
    print(f"Driver: {res['driver']}")
    print(f"Cross-projected Invariants: {res['cross_projected_invariants']}")
    print(f"New Rotor Angle: {res['new_rotor_angle']:.4f} rad")
    print(f"Equilibrium Delta: {res['equilibrium_delta']:.4f}")
    print(f"Remaining Residual Tension: {res['remaining_residual_tension']:.4f}")

    assert res["remaining_residual_tension"] < 1.5, "Residual tension failed to discharge during play!"
    print("✓ Spontaneous internal play verification PASSED.")

def test_5_world_friction_and_lens_self_rewiring():
    print("\n--- [Test 5] World Friction & Resonance Calibration (실재 마찰 및 공진) ---")
    engine = PhaseTopologicalReconstructionEngine(dimension=16)

    # 1. Recall an internal wave
    engine.recall_memory_resonance("Apple")

    # 2. Case A: Matching External Raw Wave -> High Resonance, Low Friction
    matching_wave = engine.virtual_wave.copy()
    res_match = engine.clash_with_world_and_calibrate(matching_wave)
    print(f"[Matching Wave] Friction V_t: {res_match['phase_friction_V_t']:.4f}, Resonance Score: {res_match['resonance_score']:.4f}, Lens Rewired: {res_match['lens_self_rewired']}")

    # 3. Case B: Clashing External Raw Wave -> High Friction, Triggers Lens S_t Self-Rewiring
    clashing_wave = -1.0 * matching_wave + np.random.randn(16).astype(np.float32)
    res_clash = engine.clash_with_world_and_calibrate(clashing_wave)
    print(f"[Clashing Wave] Friction V_t: {res_clash['phase_friction_V_t']:.4f}, Resonance Score: {res_clash['resonance_score']:.4f}, Lens Rewired: {res_clash['lens_self_rewired']}")

    assert res_clash["lens_self_rewired"] == True, "High world friction failed to trigger Lens S_t Self-Rewiring!"
    print("✓ World friction and lens self-rewiring verification PASSED.")

def test_6_deferred_integration_4_stage_pipeline():
    print("\n--- [Test 6] Deferred Integration (사후 재통합 4단계 프로세스) ---")
    engine = PhaseTopologicalReconstructionEngine(dimension=16, v_critical=80.0, kappa=0.06, gamma=0.08)

    # 1단계: 감당 불능 마찰 감지와 격리 (Isolation)
    # High-friction raw wave (opposite phase to core_phase_vector)
    high_friction_wave = -1.0 * engine.core_phase_vector
    high_friction_wave[1] = 0.5
    res_isolation = engine.process_external_wave(high_friction_wave)

    print(f"Stage 1 Isolation Status: {res_isolation['status']}")
    print(f"Friction: {res_isolation['friction']:.2f}")

    assert res_isolation["status"] == "SEALED", "High friction wave was not properly isolated!"
    assert len(engine.sealed_attractors) == 1, "Sealed attractor count should be 1!"
    sealed = engine.sealed_attractors[0]
    assert sealed.is_sealed == True, "Attractor should initially be sealed!"
    print(f"Min required capacity: {sealed.min_required_capacity:.2f}")
    print("✓ Stage 1 (Isolation) verification PASSED.")

    # 2단계: 내적 놀이 상태에서의 스캔 (Scan & Trigger)
    # System capacity C(t) is initially low (0.1) -> min_required_capacity is higher (~4.73)
    res_play_low = engine.run_spontaneous_internal_play()
    print(f"Stage 2 Scan under Low Capacity: Deferred Triggered = {res_play_low['deferred_integrations_triggered']}")
    assert res_play_low['deferred_integrations_triggered'] == 0, "Deferred integration should not trigger when capacity is insufficient!"

    # System grows and lens capacity expands (C_lens >= min_required_capacity)
    engine.expand_lens_capacity(5.0)  # C_lens -> 5.1 >= min_required_capacity (~4.73)
    print(f"Expanded Lens Capacity C(t): {engine.lens_capacity:.2f}")

    # 3단계 & 4단계: 단계적 감쇄, 위상 정렬 및 사후 공진/위상 재선로화 (Phase Alignment & Resonance Limit)
    # Run play loop steps until deferred integration dynamics converge friction E(V_t) -> 0
    reintegrated_indices = []
    for step in range(500):
        res_play = engine.run_spontaneous_internal_play()
        if res_play["deferred_integrations_triggered"] > 0:
            reintegrated_indices.append(step)
            break

    print(f"Stage 3 & 4 Convergence Step: {reintegrated_indices}")
    assert len(reintegrated_indices) > 0, "Deferred integration failed to converge and reintegrate!"
    assert sealed.is_sealed == False, "Sealed attractor flag should be set to False after reintegration!"
    assert len(engine.reintegrated_invariants) == 1, "Reintegrated invariant should be absorbed into core terrain!"
    assert "Reintegrated_Ic_0" in engine.invariants, "Inference invariant Ic should be stored in invariant library!"

    print(f"Reintegrated Invariant Ic Name: 'Reintegrated_Ic_0'")
    print(f"Final Attractor Friction: {sealed.current_friction:.6f}")
    print(f"Final Attractor Delta Theta: {sealed.current_delta_theta:.6f} rad")
    assert sealed.current_friction < 0.01, "Final friction must converge to zero (< 0.01)!"
    assert abs(sealed.current_delta_theta) < 0.05, "Final phase mismatch must align (< 0.05 rad)!"

    print("✓ Stage 2, 3, 4 (Scan, Alignment, Resonance & Ic Invariant Absorption) verification PASSED.")

def run_all_verifications():
    print("=================================================================")
    print(" VERIFYING OPEN PHASE RESONATOR (PhaseTopologicalReconstruction) ")
    print("=================================================================")
    test_1_memory_re_resonance()
    test_2_imagination_superposition()
    test_3_conversation_bandwidth_restrictor()
    test_4_spontaneous_internal_play()
    test_5_world_friction_and_lens_self_rewiring()
    test_6_deferred_integration_4_stage_pipeline()
    print("\n=================================================================")
    print(" ALL 6 PHASE TOPOLOGICAL MECHANISMS VERIFIED SUCCESSFULLY! ")
    print("=================================================================")

if __name__ == "__main__":
    run_all_verifications()
