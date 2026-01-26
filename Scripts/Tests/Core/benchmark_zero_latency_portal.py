"""
BENCHMARK: Zero-Latency Portal (Step 3)
=======================================
Comparing Standard mmap (Merkaba) vs CUDA-Streamed Pinned Memory (Zero-Latency).
"""

import time
import numpy as np
import logging
import sys
import os
from numba import cuda

# Add root to path
sys.path.append(os.getcwd())

from Core.L6_Structure.Merkaba.portal import MerkabaPortal
from Core.L6_Structure.System.Metabolism.zero_latency_portal import ZeroLatencyPortal
from Core.L6_Structure.Nature.metal_field_bridge import MetalFieldBridge

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Benchmark.Portal")

def run_benchmark(iterations=5000):
    fossil_path = r"C:\Users\USER\.ollama\models\blobs\sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515"
    if not os.path.exists(fossil_path):
        logger.error(f"❌ Fossil not found for benchmark!")
        return

    chunk_size = 64 * 1024 # 64KB
    logger.info(f"🚀 Commencing 'Zero-Latency' Benchmark with {iterations} spikes...")

    # Initialize Metal Field
    field_bridge = MetalFieldBridge(size=64)
    
    # 1. TRADITIONAL MERKABA PORTAL (mmap)
    logger.info("🏺 Testing MerkabaPortal (Standard mmap)...")
    start_py = time.perf_counter()
    with MerkabaPortal(fossil_path) as portal:
        for i in range(iterations):
            offset = (i * chunk_size) % (portal.file_size - chunk_size)
            data = portal.read_view(offset, chunk_size, dtype=np.uint8)
            # Simulated analysis overhead
            _ = np.mean(data)
    end_py = time.perf_counter()
    duration_py = end_py - start_py
    logger.info(f"🏺 Traditional Duration: {duration_py:.4f} seconds")

    # 2. ZERO-LATENCY PORTAL (CUDA Streaming)
    logger.info("⚡ Testing ZeroLatencyPortal (Pinned Streaming)...")
    start_metal = time.perf_counter()
    with ZeroLatencyPortal(fossil_path) as z_portal:
        # Pinned buffer initialization
        z_portal.stream_to_metal(0, chunk_size, dtype=np.uint8)
        
        for i in range(iterations):
            offset = (i * chunk_size) % (z_portal.file_size - chunk_size)
            data = z_portal.stream_to_metal(offset, chunk_size, dtype=np.uint8)
            # Simulated analysis overhead (GPU would do this in real scenario)
            _ = np.mean(data)
    end_metal = time.perf_counter()
    duration_metal = end_metal - start_metal
    logger.info(f"⚡ Zero-Latency Duration: {duration_metal:.4f} seconds")

    # 3. RESULTS
    speedup = duration_py / duration_metal if duration_metal > 0 else float('inf')
    logger.info("\n--- [BENCHMARK RESULTS] ---")
    logger.info(f"  - Speedup Factor: {speedup:.1f}x faster")
    logger.info(f"  - Traditional Avg: {duration_py*1000/iterations:.4f}ms per chunk")
    logger.info(f"  - Zero-Latency Avg: {duration_metal*1000/iterations:.4f}ms per chunk")
    logger.info("---------------------------\n")

    if speedup > 1.2: # For I/O, even 20% is significant for giant files
        logger.info("🏆 The Zero-Latency Portal is open. The arteries are flowing.")
    else:
        logger.warning("⚠️ Speedup is marginal. OS cache might be masking the raw NVMe speed.")

if __name__ == "__main__":
    run_benchmark(iterations = 3000)
