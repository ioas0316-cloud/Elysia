import sys
import os
import time
import math
import psutil
import statistics
import json

# Add root folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.fractal_rotor import Rotor

def test_retrocausal_imitation_limit():
    print("\n--- 2. Retrocausal Imitation Limit: Phase Noise Injection ---")
    results = []

    # We will build a simple 2-level system
    root = Rotor("Galaxy", level=0)
    planet1 = Rotor("Planet1", level=1, parent=root)
    planet2 = Rotor("Planet2", level=1, parent=root)
    root.attach_child(planet1)
    root.attach_child(planet2)

    # Strong coupling to try and maintain order
    root.coupling_map[(planet1.id, planet2.id)] = 1.0
    root.coupling_map[(planet2.id, planet1.id)] = 1.0

    noise_levels = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    for noise_magnitude in noise_levels:
        root.phase_offset = 0.0
        planet1.phase_offset = 0.0
        planet2.phase_offset = 0.0

        tensions = []
        phase_diffs = []
        divergence_detected = False
        hunting_detected = False

        for tick in range(100):
            # Inject noise (Retrocausal wave noise simulation)
            import random
            noise1 = random.uniform(-noise_magnitude, noise_magnitude)
            noise2 = random.uniform(-noise_magnitude, noise_magnitude)

            planet1.phase_offset += noise1
            planet2.phase_offset += noise2

            # Use a tiny delta rotation to observe
            root.observe(0.05)

            # Measure directly the stored tension
            current_tension = max(planet1.tension, planet2.tension)
            tensions.append(current_tension)

            # Calculate phase diff (wrap around pi)
            diff = abs(planet1.current_phase - planet2.current_phase)
            if diff > math.pi:
                diff = 2*math.pi - diff
            phase_diffs.append(diff)

            # Tension limit in fractal_rotor is (math.pi / 2.0) / (level + 1)
            # Level 1 limit is ~0.78 rad
            if current_tension > planet1.tension_limit:
                 divergence_detected = True

        avg_tension = statistics.mean(tensions)
        max_tension = max(tensions)
        avg_phase_diff = statistics.mean(phase_diffs)

        print(f"Noise {noise_magnitude:4.1f} rad -> Avg Tension: {avg_tension:.4f} | Max Tension: {max_tension:.4f} | Avg Phase Diff: {avg_phase_diff:.4f} rad | Diverged: {divergence_detected}")

if __name__ == "__main__":
    test_retrocausal_imitation_limit()
