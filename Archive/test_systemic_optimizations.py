import time
import os
from pathlib import Path
from Core.S1_Body.L1_Foundation.Hardware.somatic_ssd import SomaticSSD
from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

def benchmark_somatic_ssd():
    print("--- Sub-Somatic Benchmark ---")
    soma = SomaticSSD()
    
    # Measure cached pulse (O(1))
    start = time.perf_counter()
    for _ in range(100):
        soma.proprioception(throttle=2.0)
    end = time.perf_counter()
    avg_cached = (end - start) / 100
    print(f"Average Cached Pulse (O(1)): {avg_cached*1000:.4f} ms")
    
    # Force a scan (O(N))
    start = time.perf_counter()
    soma.proprioception(throttle=-1.0)
    end = time.perf_counter()
    scan_time = end - start
    print(f"Force Scan Time (O(N)): {scan_time*1000:.2f} ms")
    print(f"Speedup: {scan_time / (avg_cached + 1e-9):.1f}x")

def benchmark_logos_bridge():
    print("\n--- LogosBridge Vectorized Search Benchmark ---")
    LogosBridge.polymerize_spectrum()
    test_vec = SovereignVector([0.1]*21)
    
    # Warm up
    LogosBridge.find_closest_concept(test_vec)
    
    start = time.perf_counter()
    iters = 500
    for _ in range(iters):
        LogosBridge.find_closest_concept(test_vec)
    end = time.perf_counter()
    avg_search = (end - start) / iters
    print(f"Average Vector Search (Vectorized): {avg_search*1000:.4f} ms")

if __name__ == "__main__":
    benchmark_somatic_ssd()
    benchmark_logos_bridge()
