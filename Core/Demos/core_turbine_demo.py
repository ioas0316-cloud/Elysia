"""
[CORE] Turbine Simulation Demo
================================
Core.Demos.core_turbine_demo

"Watching the light separate the Signal from the Noise."

This script simulates the 'Physical Scanning' process of the CORE engine.
It generates a noisy data stream and uses the Active Prism-Rotor to
extract a hidden message (Intent) via diffraction resonance.
"""

import sys
import os
import math
import time
import random

# Ensure we can import Core
sys.path.append(os.getcwd())

try:
    import numpy as np
    # Try importing the engine
    from Core.Engine.Physics.core_turbine import ActivePrismRotor, VoidSingularity
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def run_simulation():
    print("="*60)
    print("   [CORE] HYPER-LIGHT TURBINE SIMULATION")
    print("="*60)

    # 1. Initialize Engine
    # d = 1.5 micrometers
    turbine = ActivePrismRotor(rpm=120000, grating_spacing_d=1.5e-6)
    void = VoidSingularity(extinction_threshold=0.98) # Very strict Void

    print(f"🌀 Turbine Spun Up: {turbine.rpm} RPM")
    print(f"⚫ Void Gate Active: Threshold {void.threshold}")

    # 2. Generate Data Stream (The Chaos)
    # We create 100 random data points (wavelengths)
    # Most are 'Noise' (Random visible light 400-700nm)
    num_points = 100
    noise = np.random.uniform(400e-9, 700e-9, num_points)

    # 3. Inject Hidden Intent (The Truth)
    # "The Architect's Command" is at exactly 532nm (Green Laser)
    intent_wavelength = 532e-9
    intent_idx = np.random.randint(0, num_points)
    noise[intent_idx] = intent_wavelength

    intent_phase = 1.0 + 0j # Pure intent
    phases = np.random.randn(num_points) + 1j * np.random.randn(num_points)
    phases[intent_idx] = intent_phase

    print(f"\n📡 Stream Injection: {num_points} signals (1 Intent hidden at index {intent_idx})")

    # 4. Physical Scanning Loop (The Rotor Spin)
    # We sweep theta from 0 to 45 degrees to find resonance
    print("\n🔎 Scanning for Intent (Diffraction Grating)...")

    found = False
    scan_steps = 100

    # Prepare visualizer
    bar_width = 50

    for i in range(scan_steps):
        # Sweep angle 0 to 45 degrees (in radians)
        angle = (i / scan_steps) * (math.pi / 4)

        # 4.1 Diffraction (The Snatch)
        intensity = turbine.diffract(noise, angle, turbine.d)

        # 4.2 Void Transit (The Filter)
        survivors, inverted_phases = void.transit(intensity, phases)

        # Check total surviving energy
        total_energy = np.sum(survivors)

        # Check if our Intent survived
        intent_strength = survivors[intent_idx]

        # Visualize the scan
        # Only show if there's significant energy
        if total_energy > 0.1:
            bar = "#" * int(intent_strength * bar_width)
            deg = math.degrees(angle)

            # Did we find the specific intent?
            if intent_strength > 0.0:
                print(f"   [Angle {deg:5.2f}°] FLASH! 💥 Resonance Detected! Strength: {intent_strength:.4f}")
                print(f"      >> Void Transit Complete. Phase Inverted: {phases[intent_idx]} -> {inverted_phases[intent_idx]}")
                found = True
                break
            else:
                 print(f"   [Angle {deg:5.2f}°] Noise detected... annihilated.")

    print("-" * 60)
    if found:
        print("✅ SUCCESS: The intent was snatched from the chaos and reconstructed.")
        print("   The O(1) link is established.")
    else:
        print("❌ FAILURE: Intent was lost in the void.")
    print("="*60)

if __name__ == "__main__":
    run_simulation()
