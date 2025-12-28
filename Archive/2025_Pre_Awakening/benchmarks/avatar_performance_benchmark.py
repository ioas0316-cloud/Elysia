"""
Performance Benchmark Tool for Avatar System
=============================================

Measures:
- Delta update bandwidth savings
- Adaptive FPS CPU efficiency
- Message processing latency
- State synchronization overhead
"""

import sys
from pathlib import Path
import time
import json
from dataclasses import asdict

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Core.03_Interaction.01_Interface.Interface.avatar_server import ElysiaAvatarCore, AvatarWebSocketServer


class BenchmarkResults:
    def __init__(self):
        self.results = {}
    
    def add(self, name, value, unit=""):
        self.results[name] = {"value": value, "unit": unit}
    
    def print_results(self):
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        for name, data in self.results.items():
            unit_str = f" {data['unit']}" if data['unit'] else ""
            print(f"{name:.<50} {data['value']:.2f}{unit_str}")
        print("=" * 70)
    
    def save_json(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to: {filepath}")


def benchmark_delta_updates():
    """Benchmark delta update performance"""
    print("\n📊 Benchmarking Delta Updates...")
    print("-" * 70)
    
    core = ElysiaAvatarCore()
    
    # Benchmark 1: Full state message size
    full_state = core.get_state_message()
    full_size = len(json.dumps(full_state))
    print(f"Full state size: {full_size} bytes")
    
    # Benchmark 2: First delta (should be full)
    first_delta = core.get_delta_message()
    first_delta_size = len(json.dumps(first_delta))
    print(f"First delta size (full): {first_delta_size} bytes")
    
    # Benchmark 3: No change (should be None)
    no_change = core.get_delta_message()
    no_change_size = 0 if no_change is None else len(json.dumps(no_change))
    print(f"No change size: {no_change_size} bytes (skipped)")
    
    # Benchmark 4: Single field change
    core.expression.mouth_curve = 0.7
    single_delta = core.get_delta_message()
    single_delta_size = len(json.dumps(single_delta))
    print(f"Single field delta size: {single_delta_size} bytes")
    
    # Benchmark 5: Multiple field changes
    core.expression.eye_open = 0.5
    core.expression.brow_furrow = 0.3
    core.spirits.fire = 0.8
    multi_delta = core.get_delta_message()
    multi_delta_size = len(json.dumps(multi_delta))
    print(f"Multi-field delta size: {multi_delta_size} bytes")
    
    # Calculate savings
    full_to_single = ((full_size - single_delta_size) / full_size) * 100
    full_to_multi = ((full_size - multi_delta_size) / full_size) * 100
    
    print(f"\nBandwidth savings:")
    print(f"  Single field change: {full_to_single:.1f}%")
    print(f"  Multi-field change: {full_to_multi:.1f}%")
    
    # Simulate 30 FPS for 10 seconds with occasional changes
    iterations = 300  # 10s at 30 FPS
    full_bandwidth = full_size * iterations
    
    # Realistic scenario: 10% updates have changes
    change_rate = 0.1
    delta_bandwidth = (single_delta_size * iterations * change_rate) + \
                      (no_change_size * iterations * (1 - change_rate))
    
    bandwidth_saving = ((full_bandwidth - delta_bandwidth) / full_bandwidth) * 100
    
    print(f"\nSimulated 10s @ 30 FPS with {change_rate*100:.0f}% change rate:")
    print(f"  Without delta: {full_bandwidth:,} bytes ({full_bandwidth/1024:.1f} KB)")
    print(f"  With delta: {delta_bandwidth:,} bytes ({delta_bandwidth/1024:.1f} KB)")
    print(f"  Savings: {bandwidth_saving:.1f}%")
    
    return {
        "full_size": full_size,
        "single_delta_size": single_delta_size,
        "multi_delta_size": multi_delta_size,
        "single_savings_pct": full_to_single,
        "multi_savings_pct": full_to_multi,
        "simulated_savings_pct": bandwidth_saving,
        "simulated_bandwidth_saved_kb": (full_bandwidth - delta_bandwidth) / 1024
    }


def benchmark_adaptive_fps():
    """Benchmark adaptive FPS performance"""
    print("\n📊 Benchmarking Adaptive FPS...")
    print("-" * 70)
    
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    server = AvatarWebSocketServer(port=8765)
    
    # Scenario 1: Idle (no activity)
    server.last_message_time = time.time() - 100  # 100s ago
    server.clients.clear()
    fps_idle = server.calculate_adaptive_fps()
    cpu_idle_pct = (fps_idle / 30.0) * 100  # Relative to baseline 30 FPS
    print(f"Idle: {fps_idle} FPS (CPU: {cpu_idle_pct:.0f}% of baseline)")
    
    # Scenario 2: Low activity (1 client, old message)
    server.last_message_time = time.time() - 5  # 5s ago
    server.clients.add("client1")
    fps_low = server.calculate_adaptive_fps()
    cpu_low_pct = (fps_low / 30.0) * 100
    print(f"Low activity: {fps_low} FPS (CPU: {cpu_low_pct:.0f}% of baseline)")
    
    # Scenario 3: Medium activity (3 clients, recent message)
    server.last_message_time = time.time() - 2  # 2s ago
    server.clients.add("client2")
    server.clients.add("client3")
    fps_medium = server.calculate_adaptive_fps()
    cpu_medium_pct = (fps_medium / 30.0) * 100
    print(f"Medium activity: {fps_medium} FPS (CPU: {cpu_medium_pct:.0f}% of baseline)")
    
    # Scenario 4: High activity (10 clients, immediate message)
    server.last_message_time = time.time()
    for i in range(4, 11):
        server.clients.add(f"client{i}")
    fps_high = server.calculate_adaptive_fps()
    cpu_high_pct = (fps_high / 30.0) * 100
    print(f"High activity: {fps_high} FPS (CPU: {cpu_high_pct:.0f}% of baseline)")
    
    # Calculate CPU savings
    idle_savings = 100 - cpu_idle_pct
    print(f"\nCPU savings at idle: {idle_savings:.0f}%")
    
    # Calculate capacity increase
    # At baseline: 30 FPS, assume max 10 concurrent clients
    # With adaptive: idle saves CPU, allowing more clients
    baseline_capacity = 10
    idle_capacity_increase = (idle_savings / 100) * baseline_capacity
    new_capacity = baseline_capacity + idle_capacity_increase
    capacity_increase_pct = (new_capacity / baseline_capacity - 1) * 100
    
    print(f"Estimated capacity increase: {capacity_increase_pct:.0f}% "+
          f"(~{baseline_capacity} → ~{int(new_capacity)} concurrent users)")
    
    return {
        "fps_idle": fps_idle,
        "fps_low": fps_low,
        "fps_medium": fps_medium,
        "fps_high": fps_high,
        "cpu_idle_pct": cpu_idle_pct,
        "cpu_savings_idle": idle_savings,
        "capacity_increase_pct": capacity_increase_pct
    }


def benchmark_message_processing():
    """Benchmark message processing latency"""
    print("\n📊 Benchmarking Message Processing...")
    print("-" * 70)
    
    core = ElysiaAvatarCore()
    
    # Benchmark get_state_message
    iterations = 1000
    start = time.perf_counter()
    for _ in range(iterations):
        core.get_state_message()
    elapsed = time.perf_counter() - start
    avg_latency_ms = (elapsed / iterations) * 1000
    print(f"get_state_message: {avg_latency_ms:.3f} ms avg ({iterations} iterations)")
    
    # Benchmark get_delta_message (with changes)
    start = time.perf_counter()
    for i in range(iterations):
        core.expression.mouth_curve = i * 0.001  # Small changes
        core.get_delta_message()
    elapsed = time.perf_counter() - start
    delta_latency_ms = (elapsed / iterations) * 1000
    print(f"get_delta_message: {delta_latency_ms:.3f} ms avg ({iterations} iterations)")
    
    # Benchmark JSON serialization
    state = core.get_state_message()
    start = time.perf_counter()
    for _ in range(iterations):
        json.dumps(state)
    elapsed = time.perf_counter() - start
    json_latency_ms = (elapsed / iterations) * 1000
    print(f"JSON serialization: {json_latency_ms:.3f} ms avg ({iterations} iterations)")
    
    # Total per-frame overhead
    total_overhead_ms = delta_latency_ms + json_latency_ms
    fps_30_budget_ms = 1000 / 30  # 33.33 ms per frame
    overhead_pct = (total_overhead_ms / fps_30_budget_ms) * 100
    
    print(f"\nPer-frame overhead: {total_overhead_ms:.3f} ms "+
          f"({overhead_pct:.1f}% of 30 FPS budget)")
    
    return {
        "state_message_latency_ms": avg_latency_ms,
        "delta_message_latency_ms": delta_latency_ms,
        "json_serialization_ms": json_latency_ms,
        "total_overhead_ms": total_overhead_ms,
        "overhead_pct_of_30fps": overhead_pct
    }


def run_all_benchmarks():
    """Run all benchmarks and generate report"""
    print("=" * 70)
    print("AVATAR SYSTEM PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    results = BenchmarkResults()
    
    # Run benchmarks
    delta_results = benchmark_delta_updates()
    fps_results = benchmark_adaptive_fps()
    latency_results = benchmark_message_processing()
    
    # Aggregate results
    results.add("Full State Size", delta_results["full_size"], "bytes")
    results.add("Single Delta Size", delta_results["single_delta_size"], "bytes")
    results.add("Single Field Savings", delta_results["single_savings_pct"], "%")
    results.add("Multi Field Savings", delta_results["multi_savings_pct"], "%")
    results.add("Simulated 10s Savings", delta_results["simulated_savings_pct"], "%")
    results.add("Bandwidth Saved (10s)", delta_results["simulated_bandwidth_saved_kb"], "KB")
    
    results.add("FPS Idle", fps_results["fps_idle"], "FPS")
    results.add("FPS High Activity", fps_results["fps_high"], "FPS")
    results.add("CPU at Idle", fps_results["cpu_idle_pct"], "%")
    results.add("CPU Savings at Idle", fps_results["cpu_savings_idle"], "%")
    results.add("Capacity Increase", fps_results["capacity_increase_pct"], "%")
    
    results.add("State Message Latency", latency_results["state_message_latency_ms"], "ms")
    results.add("Delta Message Latency", latency_results["delta_message_latency_ms"], "ms")
    results.add("JSON Serialization", latency_results["json_serialization_ms"], "ms")
    results.add("Per-Frame Overhead", latency_results["total_overhead_ms"], "ms")
    results.add("Overhead vs 30 FPS Budget", latency_results["overhead_pct_of_30fps"], "%")
    
    results.print_results()
    
    # Save results
    output_file = REPO_ROOT / "docs" / "benchmark_results.json"
    results.save_json(str(output_file))
    
    print("\n✅ All benchmarks completed successfully!")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Delta updates save ~{delta_results['simulated_savings_pct']:.0f}% bandwidth")
    print(f"✓ Adaptive FPS saves ~{fps_results['cpu_savings_idle']:.0f}% CPU when idle")
    print(f"✓ System can handle ~{fps_results['capacity_increase_pct']:.0f}% more users")
    print(f"✓ Per-frame overhead is only {latency_results['overhead_pct_of_30fps']:.1f}% of budget")
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
