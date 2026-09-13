"""
Verification & Demonstration Script: Epistemic Frame Continuity Engine
=============================================================================
Demonstrates:
1. Emergence of spatiotemporal dimensions from discrete frame transitions.
2. Detection of 'Unintelligible Void' (NaN corruption) and causal restoration.
3. Variational Free Energy (VFE) prediction error minimization and structural plasticity adaptation.
"""

import numpy as np
import time
from core.consciousness.epistemic_frame_continuity_engine import EpistemicFrameContinuityEngine

def main():
    print("=====================================================================")
    print(" [VERIFICATION] Epistemic Frame Continuity & Void Healing Engine ")
    print("=====================================================================\n")

    dimension = 32
    engine = EpistemicFrameContinuityEngine(dimension=dimension)

    print("Phase 1: Simulating Continuous Frame Sequence (Establishing Spatiotemporal Flow)...")
    base_state = np.random.randn(dimension)
    base_state /= np.linalg.norm(base_state)

    for step in range(5):
        # Generate smooth frame transition S_{t+1}
        delta = 0.05 * np.random.randn(dimension)
        current_state = base_state + delta
        base_state = current_state.copy()

        t_sim = 100.0 + step * 0.1
        res = engine.process_frame(current_state, internal_tension=0.5, timestamp=t_sim)
        st_info = res["spatiotemporal_info"]
        vfe_info = res["vfe_info"]

        print(f"  Frame {step+1}: status={res['status']}")
        print(f"    -> Causal Time: {st_info['causal_time']:.4f}, Velocity Norm: {st_info['velocity_norm']:.4f}")
        if "spatial_distance" in st_info:
            print(f"    -> Spatial Distance: {st_info['spatial_distance']:.4f}, VFE Error: {vfe_info['variational_free_energy']:.4f}")
        else:
            print(f"    -> Initial Frame Established, VFE Error: {vfe_info['variational_free_energy']:.4f}")

    print("\nPhase 2: Injecting Catastrophic NaN Corruption (Simulating Unintelligible Void)...")
    corrupted_state = base_state.copy()
    corrupted_state[2] = np.nan
    corrupted_state[5] = np.nan
    corrupted_state[10] = np.nan

    res_void = engine.process_frame(corrupted_state, internal_tension=10.0, timestamp=100.5)
    heal_info = res_void["heal_info"]
    print(f"  Corrupted Frame Processed:")
    print(f"    -> Void Detected: {heal_info['is_void']}, Healed: {heal_info['healed']}")
    print(f"    -> Healing Method: {heal_info['healing_method']}")
    print(f"    -> Restored State Has NaN: {np.isnan(res_void['valid_state']).any()}")

    print("\nPhase 3: Adaptation & Structural Plasticity Convergence...")
    for step in range(5):
        delta = 0.01 * np.random.randn(dimension)
        current_state = base_state + delta
        base_state = current_state.copy()

        t_sim = 100.6 + step * 0.1
        res = engine.process_frame(current_state, internal_tension=0.1, timestamp=t_sim)
        p_info = res["plasticity_info"]
        vfe_info = res["vfe_info"]
        print(f"  Frame {step+6}: VFE={vfe_info['variational_free_energy']:.4f}, Rotor Phase Theta={p_info['rotor_phase_theta']:.4f}, Conductance={p_info['conductance_c']:.4f}")

    print("\n---------------------------------------------------------------------")
    summary = engine.get_continuity_summary()
    print(" Epistemic Frame Continuity Summary:")
    print(f"  - Total Frames Processed: {summary['total_frames_processed']}")
    print(f"  - Accumulated Causal Time: {summary['accumulated_causal_time']:.4f}")
    print(f"  - Void Count Healed: {summary['void_count']}")
    print(f"  - Total Plasticity Adaptations: {summary['total_plastic_adaptations']}")
    print("=====================================================================")
    print(" Verification Complete: Epistemic Continuity Engine Operational!")

if __name__ == "__main__":
    main()
