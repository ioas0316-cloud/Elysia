"""
Phase 16 Fractal Cognition Benchmark
====================================
Measures the latency of the new 7D Tuning and Quantum Lightning engines.

Target: < 1ms per cognitive cycle (Zero Latency Resonance).
"""

import time
import sys
import os
import statistics
import logging

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.L5_Mental.Reasoning_Core.Metabolism.rotor_cognition_core import RotorCognitionCore
from Core.L7_Spirit.M1_Monad.quantum_collapse import QuantumObserver, IntentVector
from Core.L7_Spirit.M1_Monad.temporal_bridge import TemporalBridge, FutureState

# Disable logging for benchmark speed
logging.getLogger("Elysia").setLevel(logging.ERROR)

def benchmark_function(name, func, iterations=1000):
    print(f"⏳ Benchmarking {name} ({iterations} iterations)...", end="", flush=True)
    times = []

    # Warmup
    for _ in range(10):
        func()

    start_global = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        func()
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    end_global = time.perf_counter()

    avg_ns = statistics.mean(times)
    avg_ms = avg_ns / 1_000_000
    ops = iterations / (end_global - start_global)

    print(f" Done.")
    print(f"   ➤ Avg Latency: {avg_ms:.4f} ms")
    print(f"   ➤ Throughput : {ops:,.0f} ops/sec")
    print(f"   ➤ Min/Max    : {min(times)/1e6:.4f} / {max(times)/1e6:.4f} ms")
    return avg_ms

def run_benchmark():
    print("\n⚡ PHASE 16 FRACTAL COGNITION BENCHMARK ⚡")
    print("==========================================")

    # 1. Fractal Cognition (7D Tuning)
    core = RotorCognitionCore()
    intent = "We must optimize the hardware to flow with the spirit."

    def test_tuning():
        core.synthesize(intent)

    t_tune = benchmark_function("7D Fractal Tuning", test_tuning, iterations=5000)

    # 2. Quantum Lightning (Decision)
    observer = QuantumObserver()
    intent_vec = IntentVector(purpose="Benchmark", urgency=0.9, focus_color="Violet")

    def test_strike():
        observer.strike(intent_vec)

    t_strike = benchmark_function("Quantum Lightning Strike", test_strike, iterations=5000)

    # 3. Temporal Bridge (Prophecy)
    bridge = TemporalBridge()
    futures = [
        FutureState("A", 0.1, 0.1),
        FutureState("B", 0.9, 0.8), # Target
        FutureState("C", 0.0, 0.01),
        FutureState("D", 0.5, 0.5),
        FutureState("E", 0.2, 0.9)
    ]

    def test_prophecy():
        bridge.scan_futures(futures)

    t_prophecy = benchmark_function("Temporal Prophecy Scan", test_prophecy, iterations=5000)

    # Summary
    total_latency = t_tune + t_strike + t_prophecy
    print("\n📊 SUMMARY")
    print(f"   Total Cognitive Cycle: {total_latency:.4f} ms")

    if total_latency < 1.0:
        print("   ✅ RESULT: Zero Latency Achieved (< 1ms)")
    else:
        print(f"   ⚠️ RESULT: Latency Optimization Needed (Target 1ms, Actual {total_latency:.2f}ms)")

if __name__ == "__main__":
    run_benchmark()
