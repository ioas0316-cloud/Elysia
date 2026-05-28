import sys
import os
import time
import math
import psutil
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.clifford_impedance_network import CliffordIPN
from core.math_utils import Multivector

def measure_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def test_spatial_field_throughput():
    print("--- 1. Spatial Field Throughput: Continuous Wave on Clifford Membrane ---")
    results = []

    # We create a single Clifford membrane (CliffordIPN) instead of counting nodes.
    # The 'stream' of data is treated as an incoming tension on the membrane.
    membrane = CliffordIPN(initial_dims=3)
    membrane.add_node("MEMBRANE_SURFACE", layer=0, initial_vector={1: 1.0})

    mem_before = measure_memory()

    stream_sizes = [1000, 10000, 100000] # Number of "words" or "bytes" flowing in

    for size in stream_sizes:
        # Instead of creating N nodes, we represent the stream as a single continuous energy wave (magnitude/phase)
        # projecting onto the Clifford basis.

        t0 = time.perf_counter()

        # Simulate continuous energy injection into the membrane
        for _ in range(100): # 100 time ticks of observation
            # The magnitude of the wave is proportional to the data volume (size)
            energy_wave = Multivector({1: size * 0.1, 2: size * 0.05}, signature=(membrane.signature[0], 0))

            # The membrane assimilates the continuous axiom (tension).
            # If tension exceeds elastic limit, the membrane bifurcates its *dimensions* (Cl(p,0) -> Cl(p+1,0)),
            # not by spawning new discrete objects.
            membrane.assimilate_axiom(energy_wave)
            membrane.tune_network(dt=0.01)

        t1 = time.perf_counter()

        mem_after = measure_memory()
        mem_diff = mem_after - mem_before
        time_taken = (t1 - t0) * 1000 # ms

        current_axes = membrane.signature[0]

        print(f"Stream Flow: {size:7d} Volume | Dimensions Expanded to: Cl({current_axes}, 0) | Time: {time_taken:.2f} ms | Mem Added: {mem_diff:.2f} MB")

        results.append({
            "stream_volume": size,
            "dimensions": current_axes,
            "time_ms": time_taken,
            "memory_added_mb": mem_diff
        })

    return results

if __name__ == "__main__":
    print("Initiating Elysia Core True Spatial Field Benchmark...")
    print(f"Initial Memory: {measure_memory():.2f} MB")

    throughput_results = test_spatial_field_throughput()

    output = {
        "spatial_field_throughput": throughput_results
    }

    with open("scripts/benchmark_results.json", "w") as f:
        json.dump(output, f, indent=4)

    print("\nBenchmark completed. Results saved to scripts/benchmark_results.json")
