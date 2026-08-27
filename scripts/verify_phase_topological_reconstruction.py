import numpy as np
import math
from synaptic_architecture.phase_topological_reconstruction_engine import PhaseTopologicalReconstructionEngine

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

def run_all_verifications():
    print("=================================================================")
    print(" VERIFYING OPEN PHASE RESONATOR (PhaseTopologicalReconstruction) ")
    print("=================================================================")
    test_1_memory_re_resonance()
    test_2_imagination_superposition()
    test_3_conversation_bandwidth_restrictor()
    test_4_spontaneous_internal_play()
    test_5_world_friction_and_lens_self_rewiring()
    print("\n=================================================================")
    print(" ALL 5 PHASE TOPOLOGICAL MECHANISMS VERIFIED SUCCESSFULLY! ")
    print("=================================================================")

if __name__ == "__main__":
    run_all_verifications()
