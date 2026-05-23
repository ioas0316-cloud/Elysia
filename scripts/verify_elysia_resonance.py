"""
Elysia Resonance Verification Script
====================================
Tests the AC Rotor Engine's ability to transition from
0000 (Void) to 1111 (Space) based on signal quality.
"""

import numpy as np
import time
from Core.System.music_box_engine import MusicBoxEngine

def test_resonance_activation():
    engine = MusicBoxEngine()
    t_vals = np.linspace(0, 1, 1024)

    print("\n--- STAGE 1: THE VOID (NO INPUT) ---")
    audio_silent = np.zeros(1024)
    video_silent = np.zeros((64, 64, 3))
    for _ in range(5):
        res = engine.process_resonance(audio_silent, video_silent)
        print(f"Density: {res['density']:.3f} | Signature: {engine.get_bit_signature()}")

    print("\n--- STAGE 2: RESONANCE (727Hz SINE WAVE) ---")
    # 727Hz is chosen to hit low Impedance (Z) based on our L/C tuning
    audio_resonant = np.sin(2 * np.pi * 727 * t_vals)
    for _ in range(15):
        res = engine.process_resonance(audio_resonant, video_silent)
        print(f"Density: {res['density']:.3f} | Z: {res['impedance']:.3f} | Signature: {engine.get_bit_signature()}")
        time.sleep(0.05)

    print("\n--- STAGE 3: NOISE (ENTROPY RESISTANCE) ---")
    audio_noise = np.random.normal(0, 1, 1024)
    for _ in range(10):
        res = engine.process_resonance(audio_noise, video_silent)
        print(f"Density: {res['density']:.3f} | Z: {res['impedance']:.3f} | Signature: {engine.get_bit_signature()}")
        time.sleep(0.05)

if __name__ == "__main__":
    test_resonance_activation()
