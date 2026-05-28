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

def measure_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # in MB

def build_fractal_rotor_tree(depth, breadth):
    root = Rotor(id_tag="Root_Galaxy", level=0)
    current_level_nodes = [root]
    for d in range(1, depth + 1):
        next_level_nodes = []
        for parent in current_level_nodes:
            for b in range(breadth):
                child = Rotor(id_tag=f"L{d}_B{b}_{id(parent)}", level=d, parent=parent)
                parent.attach_child(child)
                next_level_nodes.append(child)
        current_level_nodes = next_level_nodes
    return root

def test_rotor_scaling():
    print("--- 1. Structural Pragmatism: Rotor Scaling Benchmark ---")
    results = []

    for depth in [1, 2, 3, 4]:
        for breadth in [2, 3, 5]:
            mem_before = measure_memory()
            root = build_fractal_rotor_tree(depth, breadth)

            # Count total nodes
            def count_nodes(r):
                return 1 + sum(count_nodes(c) for c in r.sub_rotors)
            total_nodes = count_nodes(root)

            # Measure observe time
            t0 = time.perf_counter()
            for _ in range(10):  # 10 ticks
                root.observe(global_rotation_delta=0.1)
            t1 = time.perf_counter()

            mem_after = measure_memory()
            mem_diff = mem_after - mem_before
            avg_time_per_tick = (t1 - t0) / 10.0

            # Time per node
            time_per_node_ms = (avg_time_per_tick / total_nodes) * 1000

            results.append({
                "depth": depth,
                "breadth": breadth,
                "total_nodes": total_nodes,
                "avg_time_per_tick_sec": avg_time_per_tick,
                "time_per_node_ms": time_per_node_ms,
                "memory_added_mb": mem_diff
            })

            print(f"Depth {depth}, Breadth {breadth} -> Nodes: {total_nodes:5d} | Time/tick: {avg_time_per_tick:.5f}s | Mem Added: {mem_diff:.2f}MB")

            # Clean up
            del root
            import gc
            gc.collect()

    return results

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

            root.observe(0.05)

            tensions.append(max(planet1.tension, planet2.tension))

            # Calculate phase diff (wrap around pi)
            diff = abs(planet1.current_phase - planet2.current_phase)
            if diff > math.pi:
                diff = 2*math.pi - diff
            phase_diffs.append(diff)

            if max(planet1.tension, planet2.tension) > math.pi / 2.0:
                 divergence_detected = True

            # Hunting: Rapid phase sign changes
            if tick > 5:
                if (planet1.phase_offset * tensions[-2]) < 0: # Simple oscillation heuristic
                   hunting_detected = True

        avg_tension = statistics.mean(tensions)
        max_tension = max(tensions)
        avg_phase_diff = statistics.mean(phase_diffs)

        print(f"Noise {noise_magnitude:4.1f} rad -> Avg Tension: {avg_tension:.4f} | Max Tension: {max_tension:.4f} | Avg Phase Diff: {avg_phase_diff:.4f} rad | Diverged: {divergence_detected}")

        results.append({
            "noise_magnitude": noise_magnitude,
            "avg_tension": avg_tension,
            "max_tension": max_tension,
            "divergence": divergence_detected
        })

    return results

if __name__ == "__main__":
    print("Initiating Elysia Core Benchmark Engine...")
    print(f"Initial Memory: {measure_memory():.2f} MB")

    scaling_results = test_rotor_scaling()
    noise_results = test_retrocausal_imitation_limit()

    output = {
        "scaling_benchmark": scaling_results,
        "noise_benchmark": noise_results
    }

    with open("scripts/benchmark_results.json", "w") as f:
        json.dump(output, f, indent=4)

    print("\nBenchmark completed. Results saved to scripts/benchmark_results.json")
