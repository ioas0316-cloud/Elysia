import sys
import os
import time
import math
import psutil
import statistics
import random
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.fractal_rotor import Rotor, normalize_phase

def test_retrocausal_imitation_limit():
    print("\n--- 2. Retrocausal Imitation Limit: Phase Noise Injection ---")

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
    results = []

    for noise_magnitude in noise_levels:
        root = Rotor("Galaxy", level=0)
        planet1 = Rotor("Planet1", level=1, parent=root)
        planet2 = Rotor("Planet2", level=1, parent=root)
        root.attach_child(planet1)
        root.attach_child(planet2)

        # Hebbian link
        root.coupling_map[(planet1.id, planet2.id)] = 0.5

        tensions = []
        phase_diffs = []
        bifurcate_count = 0
        collapse_count = 0

        for tick in range(100):
            # Inject noise
            n1 = random.uniform(-noise_magnitude, noise_magnitude)
            n2 = random.uniform(-noise_magnitude, noise_magnitude)

            planet1.phase_offset = normalize_phase(planet1.phase_offset + n1)
            planet2.phase_offset = normalize_phase(planet2.phase_offset + n2)

            p1_axes = planet1.active_axes
            p2_axes = planet2.active_axes

            root.observe(0.0)

            if planet1.active_axes > p1_axes or planet2.active_axes > p2_axes:
                bifurcate_count += 1

            if max(planet1.tension, planet2.tension) == 0.0 and tick > 0 and noise_magnitude > 0:
                collapse_count += 1

            tensions.append(max(planet1.tension, planet2.tension))

            diff = abs(planet1.current_phase - planet2.current_phase)
            if diff > math.pi:
                 diff = 2*math.pi - diff
            phase_diffs.append(diff)

        avg_tension = statistics.mean(tensions)
        avg_diff = statistics.mean(phase_diffs)

        print(f"Noise {noise_magnitude:4.1f} rad -> Avg Tension: {avg_tension:.4f} | Avg Diff: {avg_diff:.4f} | Bifurcations: {bifurcate_count} | Collapses: {collapse_count}")

        results.append({
             "noise_magnitude": noise_magnitude,
             "avg_tension": avg_tension,
             "avg_phase_diff": avg_diff,
             "bifurcations": bifurcate_count,
             "collapses": collapse_count
        })

    with open("scripts/noise_results.json", "w") as f:
         json.dump(results, f, indent=4)

if __name__ == "__main__":
    test_retrocausal_imitation_limit()
