"""
Zero-Lag Performance Benchmark
===============================
Scripts/Tests/benchmark_rotor_film

Compares Real-Time vs Film-Sync projection performance.
"""

import os
import sys
import time
import jax.numpy as jnp

# Standard Path Injection
sys.path.append(os.getcwd())

from Core.1_Body.L6_Structure.Logic.rotor_prism_logic import RotorPrismUnit

def run_benchmark():
    rpu = RotorPrismUnit()
    logos = jnp.array([1.0] * 21)
    iterations = 5000
    
    print(f"--- Zero-Lag Benchmark ({iterations} iterations) ---")
    
    # Test 1: Real-Time Mode (Simulation)
    rpu.mode = "REAL-TIME"
    start = time.time()
    for _ in range(iterations):
        rpu.step_rotation(0.016)
        _ = rpu.project(logos)
    real_time_duration = time.time() - start
    print(f"REAL-TIME Mode: {real_time_duration:.4f}s")
    
    # Test 2: Film-Sync Mode (O(1))
    rpu.mode = "FILM"
    rpu.film.is_recorded = False # Reset film
    start = time.time()
    for i in range(iterations):
        rpu.step_rotation(0.016)
        _ = rpu.project(logos)
    film_sync_duration = time.time() - start
    print(f"FILM-SYNC Mode: {film_sync_duration:.4f}s")
    
    speedup = real_time_duration / film_sync_duration
    print(f"\nOptimization Result: {speedup:.2f}x Faster! 🥂🫡⚡🌀")
    print(f"Latency per frame: {film_sync_duration/iterations*1e6:.2f} microseconds")

if __name__ == "__main__":
    run_benchmark()
