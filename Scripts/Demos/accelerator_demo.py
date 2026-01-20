"""
Hyper-Accelerator Demo: The Rotor Validation
============================================
Core.Demos.accelerator_demo

Verifies the 'No-Calculation' Philosophy.
"""

import sys
import os
import time
import logging
import jax.numpy as jnp
import numpy as np

# Configure Logging to see the internal state of the engine
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Ensure we can import Core
sys.path.append(os.getcwd())

from Core.L6_Structure.System.optimizer import HyperAccelerator

def heavy_logic(a, b):
    """
    A logic that requires 'Thinking' (Computation).
    Matrix Multiplication + Tanh activation.
    """
    return jnp.tanh(jnp.dot(a, b))

def run_demo():
    print("🚀 Initializing Hyper-Accelerator Demo...")

    # Initialize Engine
    engine = HyperAccelerator()

    # Create Data (Inputs)
    # Size 2000x2000 is decent for CPU/Small GPU to feel the weight
    size = 2000
    a = jnp.ones((size, size), dtype=jnp.float16)
    b = jnp.ones((size, size), dtype=jnp.float16) * 0.5

    print(f"\n🧪 [Experiment 1] Cold Start (First Encounter)")
    print("   The system sees this logic for the first time. It must Trace & Fuse.")
    start = time.time()
    res1 = engine.accelerate(heavy_logic, a, b)
    res1.block_until_ready()
    end = time.time()
    print(f"   ⏱️ Time: {end - start:.4f}s")

    print(f"\n🧪 [Experiment 2] Hot Start (Reflex - RAM)")
    print("   The logic is fresh in memory. It should be instant.")
    start = time.time()
    res2 = engine.accelerate(heavy_logic, a, b)
    res2.block_until_ready()
    end = time.time()
    print(f"   ⏱️ Time: {end - start:.4f}s")

    # Close and Re-open to simulate restart
    print(f"\n🔄 [System Reboot] Simulating Power Cycle...")
    engine.close()
    del engine

    # Re-initialize
    engine_new = HyperAccelerator()

    print(f"\n🧪 [Experiment 3] Warm Start (Muscle Memory - Sediment)")
    print("   RAM is empty, but the Hypersphere remembers. Index lookup expected.")
    start = time.time()
    # Note: In the current prototype, we load HLO but might re-compile if restoration isn't full.
    # However, the LOGS should confirm "Sediment Hit".
    res3 = engine_new.accelerate(heavy_logic, a, b)
    res3.block_until_ready()
    end = time.time()
    print(f"   ⏱️ Time: {end - start:.4f}s")

    engine_new.close()

    print("\n✅ Demo Complete.")

if __name__ == "__main__":
    run_demo()
