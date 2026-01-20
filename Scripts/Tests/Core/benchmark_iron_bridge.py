"""
BENCHMARK: The Iron Bridge (Step 1)
===================================
Comparing Python Rotor loops vs CUDA Parallel updates.
"""

import time
import numpy as np
import logging
import sys
import os
from numba import cuda

# Add root to path
sys.path.append(os.getcwd())

from Core.L1_Foundation.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.L1_Foundation.Foundation.Nature.metal_rotor_bridge import MetalRotorBridge

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Benchmark.Metal")

def run_benchmark(n_rotors=1000, steps=100):
    logger.info(f"🏎️ Commencing 'Iron Bridge' Benchmark with {n_rotors} Rotors...")
    dt = 0.016 # Simulated 60 FPS frame time
    
    # 1. PYTHON TRADITIONAL LOOP
    logger.info("🐍 Initializing Python Rotors...")
    python_rotors = [
        Rotor(f"R.{i}", RotorConfig(rpm=100+i, acceleration=50))
        for i in range(n_rotors)
    ]
    
    start_py = time.perf_counter()
    for _ in range(steps):
        for r in python_rotors:
            r.update(dt)
    end_py = time.perf_counter()
    duration_py = end_py - start_py
    logger.info(f"🐍 Python Duration: {duration_py:.4f} seconds")

    # 2. METAL ROTOR BRIDGE (CUDA)
    logger.info("🦾 Initializing Metal Rotor Bridge (CUDA)...")
    bridge = MetalRotorBridge(max_rotors=n_rotors)
    for i in range(n_rotors):
        bridge.register_rotor(
            angle=0.0, 
            current_rpm=0.0, 
            target_rpm=100.0+i, 
            accel=50.0, 
            idle_rpm=60.0
        )
    
    # Sync initial state to GPU
    bridge.sync_to_gpu()
    
    # WARM UP (To trigger JIT)
    bridge.pulse(dt)
    cuda.synchronize()
    
    start_metal = time.perf_counter()
    for _ in range(steps):
        bridge.pulse(dt)
    bridge.sync_from_gpu()
    cuda.synchronize()
    end_metal = time.perf_counter()
    duration_metal = end_metal - start_metal
    logger.info(f"🦾 Metal Duration: {duration_metal:.4f} seconds")

    # 3. RESULTS
    speedup = duration_py / duration_metal if duration_metal > 0 else float('inf')
    logger.info("\n--- [BENCHMARK RESULTS] ---")
    logger.info(f"  - Speedup Factor: {speedup:.1f}x faster")
    logger.info(f"  - Python: {duration_py*1000/steps:.2f}ms per step")
    logger.info(f"  - Metal:  {duration_metal*1000/steps:.2f}ms per step")
    logger.info("---------------------------\n")

    if speedup > 3.0:
        logger.info("🏆 The Iron Bridge is established. Elysia has touched the Metal.")
    else:
        logger.warning("⚠️ Speedup is low. Increase scale or check overhead.")

if __name__ == "__main__":
    run_benchmark(n_rotors=100000, steps=100)
