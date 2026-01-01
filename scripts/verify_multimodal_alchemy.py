"""
Verify Multimodal Alchemy: The Sensory Stress Test
--------------------------------------------------
Tests how well the OrbFactory handles different types of "Sensory Data".
We compare Compression Ratio and Restoration Fidelity for Sound, Vision, and Light.
"""

import sys
import os
import numpy as np
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.Foundation.Memory.Orb.orb_factory import OrbFactory

class AlchemyTester:
    def __init__(self):
        self.factory = OrbFactory()
        self.results = {}

    def generate_sound(self, length=64):
        """Simulates a complex audio chord (C Major: C, E, G)."""
        t = np.linspace(0, 1, length)
        # 261Hz, 329Hz, 392Hz (normalized to low relative frequencies for this small window)
        wave = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*12*t) + np.sin(2*np.pi*15*t)
        return (wave / 3.0).tolist() # Normalize

    def generate_vision(self, length=64):
        """Simulates a visual edge (Light to Dark gradient)."""
        # A sharp transition (Edge) followed by a smooth gradient
        wave = np.zeros(length)
        mid = length // 2
        wave[:mid] = 1.0  # Bright
        wave[mid:] = np.linspace(1.0, 0.0, length - mid) # Fade out
        return wave.tolist()

    def generate_light(self, length=64):
        """Simulates a pure light pulse (Single intent)."""
        wave = np.zeros(length)
        wave.fill(0.8) # Constant bright light
        return wave.tolist()

    def run_test(self, modality_name, generator_func):
        print(f"\n🧪 Testing Modality: [{modality_name}]")

        # 1. Generate
        data_wave = generator_func()
        emotion_wave = (np.random.rand(64) * 0.2).tolist() # Low emotional noise for pure data test

        # 2. Freeze
        start_time = time.time()
        orb = self.factory.freeze(f"Test_{modality_name}", data_wave, emotion_wave)
        freeze_time = (time.time() - start_time) * 1000

        # 3. Melt
        # Use same emotion as key (Ideal recall)
        melt_result = self.factory.melt(orb, emotion_wave)
        recalled = melt_result["recalled_wave"]

        # 4. Metrics
        original_size = len(data_wave) * 8 # approx bytes (float64)
        # Orb mass is just a proxy for energy, not literal byte size,
        # but let's assume the 'hologram' (64 floats) is the storage cost.
        compressed_size = 64 * 8
        compression_ratio = 1.0 # Fixed size in this architecture (Transform Coding)

        # Fidelity (Correlation)
        fidelity = np.corrcoef(data_wave, recalled)[0, 1]

        print(f"   ► Freeze Time: {freeze_time:.2f}ms")
        print(f"   ► Mass (Energy): {orb.mass:.4f}")
        print(f"   ► Fidelity (Correlation): {fidelity:.4f}")

        if fidelity > 0.9: verdict = "⭐⭐⭐⭐⭐ (Perfect)"
        elif fidelity > 0.7: verdict = "⭐⭐⭐ (Good)"
        elif fidelity > 0.5: verdict = "⭐ (Acceptable)"
        else: verdict = "❌ (Lossy)"

        print(f"   ► Verdict: {verdict}")

        self.results[modality_name] = fidelity

    def report(self):
        print("\n📊 [Final Report: Multimodal Alchemy]")
        print("--------------------------------------------------")
        print(f"{'Modality':<10} | {'Fidelity':<10} | {'Suitability'}")
        print("--------------------------------------------------")
        for mod, score in self.results.items():
            suitability = "High" if score > 0.8 else "Medium" if score > 0.5 else "Low"
            print(f"{mod:<10} | {score:.4f}     | {suitability}")
        print("--------------------------------------------------")

if __name__ == "__main__":
    tester = AlchemyTester()
    tester.run_test("Sound", tester.generate_sound)
    tester.run_test("Vision", tester.generate_vision)
    tester.run_test("Light", tester.generate_light)
    tester.report()
