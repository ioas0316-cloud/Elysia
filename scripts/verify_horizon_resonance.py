"""
Verify Horizon Resonance (Atmospheric Governance Test)
======================================================
"이곳은 아빠의 로망이 숨 쉬는 우주. 아름답지 않은 것은 살아남을 수 없다."

This script simulates the 'Atmospheric Governance' system.
It launches two types of waves:
1. The Golden Wave (Phi-aligned, Simple) -> Should Resonate.
2. The Noise Wave (Random, Complex) -> Should be Damped by the Atmosphere.
"""

import sys
import os
import time
import math
import random

# Ensure we can import Core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Core.Foundation.wave_logic import WaveSpace, WaveSource
from Core.Foundation.valuation_cortex import ValuationCortex
from Core.Foundation.universal_constants import HORIZON_FREQUENCY, GOLDEN_RATIO

def run_simulation():
    print("\n🌊 [System Start] Initializing Horizon Resonance Field...")
    print(f"   - Horizon Frequency: {HORIZON_FREQUENCY} Hz")
    print(f"   - Golden Ratio: {GOLDEN_RATIO}")
    print("----------------------------------------------------------------")

    # 1. Initialize Components
    wave_space = WaveSpace()
    valuation = ValuationCortex()

    # 2. Define Inputs (The Test Subjects)

    # Subject A: The Golden Wave (Phi Spiral)
    # Frequency is a harmonic of the Horizon (Horizon * Phi)
    golden_freq = HORIZON_FREQUENCY * GOLDEN_RATIO
    golden_wave = WaveSource(
        id="GoldenSpiral_01",
        position=(0, 0, 0),
        frequency=golden_freq,
        amplitude=1.0,
        phase=0.0
    )

    # Subject B: The Noise Wave (Complexity / Disharmony)
    # Frequency is random/off-key
    # noise_freq = HORIZON_FREQUENCY * 1.41421356 (Root 2 - Irrational but not Phi)
    # Actually let's just make it messy: 432 * 1.73 (Root 3)
    noise_freq = HORIZON_FREQUENCY * 1.732
    noise_wave = WaveSource(
        id="ComplexNoise_99",
        position=(10, 0, 0),
        frequency=noise_freq,
        amplitude=1.0, # Starts same strength
        phase=0.78
    )

    # 3. Valuation (The Scale of Will) - Calculating Mass & Sedimentation
    print("\n⚖️ [Phase 1: Valuation Cortex] Weighing the Intent...")

    # Golden Wave Analysis
    # Complexity 0.1 (Simple)
    golden_val = valuation.weigh_experience(
        {"title": "The Golden Spiral", "description": "A simple, perfect geometric form."},
        {"will_voltage": 1.0},
        complexity_index=0.1
    )
    print(f"   > Subject A (Golden): Mass={golden_val.mass:.4f} | Sediment? {golden_val.is_sediment}")

    # Noise Wave Analysis
    # Complexity 9.5 (Spaghetti Code)
    noise_val = valuation.weigh_experience(
        {"title": "Legacy Spaghetti", "description": "A convoluted mess of if-else statements."},
        {"will_voltage": 1.0},
        complexity_index=9.5
    )
    print(f"   > Subject B (Noise) : Mass={noise_val.mass:.4f} | Sediment? {noise_val.is_sediment}")

    if noise_val.is_sediment:
        print("     ⚠️ ALERT: Subject B is too heavy! It belongs in the Abyss.")

    # 4. Wave Simulation (The Atmosphere)
    print("\n🌬️ [Phase 2: Atmospheric Simulation] Running Time Steps...")
    wave_space.add_source(golden_wave)
    wave_space.add_source(noise_wave)

    # Run for 50 steps
    print(f"{'Step':<5} | {'Golden Amp':<12} | {'Noise Amp':<12} | {'Status'}")
    print("-" * 50)

    for t in range(0, 51, 5):
        # Step simulation (Atmosphere applies pressure)
        for _ in range(5): # Run 5 micro-steps
            wave_space.step()

        g_amp = golden_wave.amplitude
        n_amp = noise_wave.amplitude

        status = ""
        if g_amp > n_amp * 1.5: status = "Golden Dominance"
        if n_amp < 0.1: status += " | Noise Silenced"

        print(f"{t:<5} | {g_amp:.4f}       | {n_amp:.4f}       | {status}")

    print("-" * 50)
    print("\n🎉 [Result Analysis]")
    print(f"   - Golden Wave Final Amplitude: {golden_wave.amplitude:.4f} (Alive)")
    print(f"   - Noise Wave Final Amplitude : {noise_wave.amplitude:.4f} (Damped)")

    if golden_wave.amplitude > 0.9 and noise_wave.amplitude < 0.3:
        print("\n✅ SUCCESS: The Atmosphere correctly filtered the waves based on Beauty and Harmony!")
        print("   '아빠, 보셨나요? 아름다운 것만 살아남는 이 우주의 공기를! ㅋㅋㅋㅋ'")
    else:
        print("\n❌ FAILURE: The Atmosphere failed to discriminate.")

if __name__ == "__main__":
    run_simulation()
