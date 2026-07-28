import time
import numpy as np
import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.memory.causal_controller import CausalMemoryController
from core.intelligence.thought_field import ThoughtField
from core.intelligence.thought_element import ThoughtTransistor
from core.physics.causal_field import CausalField, InformationVoxel

def benchmark_wedge_memory():
    print("[Benchmark] Initiating Wedge Memory Performance Test...")
    controller = CausalMemoryController()

    # 1. Write scaling latency
    write_times = []
    num_engrams = 2000

    start_time = time.time()
    for i in range(num_engrams):
        t0 = time.perf_counter()
        controller.write_causal_engram(
            data_blob={"index": i, "vector": np.random.randn(12).tolist(), "concept": f"concept_node_{i}"},
            emotional_value=np.random.rand(),
            cause_id=f"cause_{i % 50}",
            origin_axis="benchmark_axis"
        )
        t1 = time.perf_counter()
        write_times.append(t1 - t0)
    total_write_time = time.time() - start_time

    avg_write_ms = np.mean(write_times) * 1000
    p99_write_ms = np.percentile(write_times, 99) * 1000

    # 2. Retrieval scaling latency (Direct XOR mmap vs standard lookup)
    # We choose 100 random keys to read
    all_keys = list(controller.index.keys())
    sample_keys = np.random.choice(all_keys, 200, replace=False)

    read_times = []
    for key in sample_keys:
        t0 = time.perf_counter()
        controller.read_engram_trace(key)
        t1 = time.perf_counter()
        read_times.append(t1 - t0)

    avg_read_ms = np.mean(read_times) * 1000
    p99_read_ms = np.percentile(read_times, 99) * 1000

    # 3. Gravitational Recall (Similarity Search across sparse spaces)
    target_vector = {"axis_mass": 0.8, "axis_entropy": 0.2, "axis_light": 0.5}
    t0 = time.perf_counter()
    recalled = controller.gravitational_recall(target_vector, initial_energy=2.0)
    t1 = time.perf_counter()
    grav_recall_ms = (t1 - t0) * 1000

    print(f" -> Write: Avg={avg_write_ms:.4f}ms, P99={p99_write_ms:.4f}ms, Total={total_write_time:.4f}s for {num_engrams} engrams")
    print(f" -> Read (Wedge O(1)): Avg={avg_read_ms:.4f}ms, P99={p99_read_ms:.4f}ms")
    print(f" -> Gravitational Recall (K-D Search equivalent): {grav_recall_ms:.4f}ms (Found {len(recalled)} nodes)")

    return {
        "num_engrams": num_engrams,
        "avg_write_ms": avg_write_ms,
        "p99_write_ms": p99_write_ms,
        "total_write_time_s": total_write_time,
        "avg_read_ms": avg_read_ms,
        "p99_read_ms": p99_read_ms,
        "grav_recall_ms": grav_recall_ms,
        "num_recalled": len(recalled)
    }

def benchmark_thought_field_dynamics():
    print("[Benchmark] Initiating Thought Field Simultaneous Conductance Solver & Plasticity Test...")
    field = ThoughtField()

    # Add nodes to form a realistic causal mesh
    num_nodes = 100
    for i in range(num_nodes):
        node = ThoughtTransistor(f"node_{i}", np.random.randn(3))
        field.add_element(node)

    # Dense coupling
    for i in range(num_nodes - 1):
        field.connect(f"node_{i}", f"node_{i+1}")
        if i % 3 == 0 and i < num_nodes - 5:
            field.connect(f"node_{i}", f"node_{i+5}")

    # Measure simultaneous energy solver (pulse)
    pulse_times = []
    for _ in range(50):
        external_inputs = {f"node_{np.random.randint(num_nodes)}": np.random.rand() * 5.0 for _ in range(5)}
        t0 = time.perf_counter()
        field.pulse(external_inputs)
        t1 = time.perf_counter()
        pulse_times.append(t1 - t0)

    avg_pulse_ms = np.mean(pulse_times) * 1000

    # Measure plasticity step (conductance reinforcement, mitosis, apoptosis, rewire)
    step_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        field.step()
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

    avg_step_ms = np.mean(step_times) * 1000

    print(f" -> Simultaneous Flow Solver (Pulse): Avg={avg_pulse_ms:.4f}ms (100 nodes, Conductance Matrix inversion)")
    print(f" -> Plasticity Step (Self-rewiring & Growth): Avg={avg_step_ms:.4f}ms")

    return {
        "num_nodes": num_nodes,
        "avg_pulse_ms": avg_pulse_ms,
        "avg_step_ms": avg_step_ms
    }

def main():
    print("=========================================================")
    print("      ELYSIA COGNITIVE & PHYSICAL CORE BENCHMARK        ")
    print("=========================================================")

    mem_results = benchmark_wedge_memory()
    field_results = benchmark_thought_field_dynamics()

    # Save results to data folder
    results_path = os.path.join("data", "topology", "benchmark_report.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    report = {
        "timestamp": time.time(),
        "memory_benchmark": mem_results,
        "thought_field_benchmark": field_results,
        "system": {
            "platform": sys.platform,
            "python_version": sys.version
        }
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"\n[Benchmark] Results saved successfully to: {results_path}")
    print("=========================================================")

if __name__ == "__main__":
    main()
