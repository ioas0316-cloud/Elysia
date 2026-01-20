"""
BENCHMARK: Heart of Metal (Step 2)
===================================
Comparing NumPy (CPU) vs CUDA (Metal) 7D Field Evolution.
"""

import time
import numpy as np
import logging
import sys
import os
from numba import cuda

# Add root to path
sys.path.append(os.getcwd())

from Core.L1_Foundation.Foundation.Nature.metal_field_bridge import MetalFieldBridge

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Benchmark.Field")

def numpy_evolve(field, dt, size, diffusion_rate):
    """NumPy implementation of the same 7D evolution logic."""
    # Laplacian (Toroidal)
    up = np.roll(field, -1, axis=0)
    down = np.roll(field, 1, axis=0)
    left = np.roll(field, -1, axis=1)
    right = np.roll(field, 1, axis=1)
    
    laplacian = (up + down + left + right - 4.0 * field)
    
    # Simple oscillation
    for d in range(7):
        osc = np.sin(dt * (d + 1)) * 0.01
        field[:, :, d] += (laplacian[:, :, d] * diffusion_rate * dt) + (osc * field[:, :, d])
    
    field *= 0.999
    return field

def run_benchmark(size=128, steps=100):
    logger.info(f"❤️ Commencing 'Heart of Metal' Benchmark with {size}x{size}x7 Field...")
    dt = 0.1
    diffusion_rate = 0.1
    
    # 1. NUMPY (CPU)
    logger.info(f"🐍 Initializing NumPy Field ({size}x{size})...")
    py_field = np.zeros((size, size, 7), dtype=np.float32)
    py_field[:, :, 0] = 0.5
    
    start_py = time.perf_counter()
    for _ in range(steps):
        py_field = numpy_evolve(py_field, dt, size, diffusion_rate)
    end_py = time.perf_counter()
    duration_py = end_py - start_py
    logger.info(f"🐍 NumPy Duration: {duration_py:.4f} seconds")

    # 2. METAL FIELD BRIDGE (CUDA)
    logger.info(f"🦾 Initializing Metal Field Bridge (CUDA)...")
    bridge = MetalFieldBridge(size=size, diffusion_rate=diffusion_rate)
    bridge.sync_to_gpu()
    
    # WARM UP
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
    logger.info(f"  - NumPy Avg: {duration_py*1000/steps:.2f}ms per pulse")
    logger.info(f"  - Metal Avg: {duration_metal*1000/steps:.2f}ms per pulse")
    logger.info("---------------------------\n")

    if speedup > 5.0:
        logger.info("🏆 The Heart of Metal is beating strong. Zero-latency Resonance achieved.")
    else:
        logger.warning("⚠️ Speedup is low. Increase field resolution or grid size.")

if __name__ == "__main__":
    # Larger size to show GPU advantage
    run_benchmark(size=256, steps=100)
