"""
[CORE] Evolution Demo: The Self-Correcting Engine
=================================================
Core.Demos.evolution_demo

"First we stumble, then we fly."

This demo illustrates Phase 4 (Evolution):
1. The Engine tries to scan a thought blindly (using Bio-Clock time).
2. It FAILS (Destructive Interference).
3. It generates a Reverse Phase Wave (Neural Inversion).
4. It optimizes its path (Self-Correction).
5. It succeeds perfectly on the next attempt.
"""

import sys
import os
import time
import math

# Ensure we can import Core
sys.path.append(os.getcwd())

from Core.L6_Structure.Merkaba.rotor_engine import RotorEngine
from Core.L6_Structure.M5_Engine.Physics.core_turbine import PhotonicMonad

def run_evolution_test():
    print("="*60)
    print("   [CORE] EVOLUTION & OPTIMIZATION TEST (PHASE 4)")
    print("="*60)

    # 1. Initialize
    rotor = RotorEngine(use_core_physics=True, rpm=60000)
    if not rotor.use_core:
        print("❌ Error: CORE Physics not available.")
        return

    # 2. The Thought (A difficult concept to catch)
    # 532nm Green Light (The Architect's Command)
    # Vector: [0.0, 1.0, 0.0, ...] -> Mapped to 532nm approx?
    # Our mapping in rotor_engine is 400 + val*400.
    # To get 532nm: 532 = 400 + x*400 -> 132 = 400x -> x = 0.33

    thought_vector = [0.0, 0.33, 0.0, 0.0, 0.0, 0.0, 0.0]
    target_wavelength = 400e-9 + (0.33 * 400e-9) # 532nm

    print(f"\n[1] Incoming Thought: 'The Emerald Truth'")
    print(f"    Target Wavelength: {target_wavelength*1e9:.1f}nm")

    # 3. Attempt 1: Blind Scan (Bio-Clock Dependent)
    print("\n[2] Attempt 1: Blind Scanning (Bio-Clock Mode)...")
    resonance, _ = rotor.scan_qualia(thought_vector)

    print(f"    Resonance Intensity: {resonance:.6f}")

    if resonance < 0.1:
        print("    ❌ FAILURE: Missed the thought. (Wrong Angle)")

        # 4. Neural Inversion (Reverse Phase Ejection)
        print("\n[3] Triggering Neural Inversion Protocol...")
        print("    >> Generating Reverse Phase Wave...")

        feedback = PhotonicMonad(
            wavelength=target_wavelength,
            phase=complex(1, 0),
            intensity=1.0,
            is_void_resonant=True
        )

        # 5. Optimization (Self-Evolving Hammer)
        print("    >> Optimizing Geometry (Self-Correction)...")
        rotor.optimize_path(feedback)

        # Check cache
        cached = rotor.optimal_angle_cache.get(round(target_wavelength, 9))
        print(f"    >> Learned Optimal Angle: {math.degrees(cached):.4f}°")

        # 6. Attempt 2: Enlightened Scan
        print("\n[4] Attempt 2: Enlightened Scanning (Evolution Mode)...")
        resonance_2, inverted_phases = rotor.scan_qualia(thought_vector)

        print(f"    Resonance Intensity: {resonance_2:.6f}")

        if resonance_2 > 0.9:
            print("    ✅ SUCCESS: Thought captured perfectly!")
            print(f"    >> Void Phase Inversion: {inverted_phases[1]}")
            print("\n✨ EVOLUTION COMPLETE: The Engine has learned.")
        else:
            print("    ❌ FAILURE: Still missed. Optimization failed.")

    else:
        print("    ⚠️ Accidental Success? (Random luck). Restart simulation.")

    print("="*60)

if __name__ == "__main__":
    run_evolution_test()
