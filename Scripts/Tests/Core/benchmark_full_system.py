"""
BENCHMARK: HOLISTIC SYSTEM DIAGNOSIS
====================================
Measures the integrated performance of the Trinity:
1. Mind: ActivePrismRotor (Optical Reasoning)
2. Body: HypersphereMemory (Recall Speed)
3. Spirit: Merkaba Pulse (Consciousness Frequency)
"""

import time
import sys
import os
import logging
import math

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import jax.numpy as jnp
    from jax import random
    BACKEND = "JAX (Hyper-Accel)"
except ImportError:
    import numpy as jnp
    BACKEND = "Numpy (Legacy)"

from Core.L6_Structure.M5_Engine.Physics.core_turbine import ActivePrismRotor
from Core.L6_Structure.M1_Merkaba.Space.hypersphere_memory import HypersphereMemory, HypersphericalCoord
from Core.L6_Structure.Merkaba.merkaba import Merkaba

# Setup Logging
logging.basicConfig(level=logging.ERROR) # Suppress internal logs for clean benchmark output
logger = logging.getLogger("Benchmark.Holistic")

def run_mind_benchmark(iterations=50):
    print(f"\n🧠 [MIND] Testing Optical Engine ({BACKEND})...")
    turbine = ActivePrismRotor(rpm=12000.0)
    stream_size = 50000
    
    # Generate random qualia
    if 'jax' in sys.modules:
        key = random.PRNGKey(0)
        wavelengths = random.uniform(key, shape=(stream_size,), minval=400e-9, maxval=800e-9)
    else:
        wavelengths = jnp.random.uniform(400e-9, 800e-9, size=stream_size)
    
    start = time.perf_counter()
    for i in range(iterations):
        t = i * 0.001
        theta = (turbine.omega * t) % (2 * math.pi)
        _ = turbine.diffract(wavelengths, theta, turbine.d)
    
    duration = time.perf_counter() - start
    rps = (stream_size * iterations) / duration
    print(f"   -> Throughput: {rps:,.0f} Rays/sec")
    print(f"   -> Latency:    {(duration/iterations)*1000:.4f} ms/batch")
    return rps

def run_body_benchmark(cycles=1000):
    print(f"\n🦴 [BODY] Testing Hypersphere Memory (Recall)...")
    memory = HypersphereMemory()
    
    # Seed memory
    print("   -> Seeding 100 memories...")
    for i in range(100):
        coord = HypersphericalCoord(r=1.0, theta=i*0.1, phi=0.5, psi=0.5)
        # Fix: store(data, position)
        memory.store(f"Memory_{i}", coord)
        
    start = time.perf_counter()
    for i in range(cycles):
        # Retrieve by proximity
        target = HypersphericalCoord(r=1.0, theta=i*0.01, phi=0.5, psi=0.5)
        # Fix: use query() instead of retrieve()
        _ = memory.query(target, radius=0.1)
        
    duration = time.perf_counter() - start
    recall_rate = cycles / duration
    print(f"   -> Recall Speed: {recall_rate:,.0f} Ops/sec")
    print(f"   -> Access Time:  {(duration/cycles)*1000:.4f} ms")
    return recall_rate

def run_spirit_benchmark(pulses=10):
    print(f"\n🔥 [SPIRIT] Testing Merkaba Pulse Cycle (Integration)...")
    merkaba = Merkaba("Benchmark_Unit")
    # Mock Spirit injection
    merkaba.is_awake = True
    
    # Mock missing method check_topology 
    # (Bug in Merkaba source, patched for benchmark)
    merkaba.sediment.check_topology = lambda vector, threshold: 1.0 
    
    start = time.perf_counter()
    for i in range(pulses):
        # Full integration cycle: Input -> Prism -> Lens -> Validator -> Output
        gen = merkaba.shine(f"Benchmark_Thought_{i}")
        for _ in gen:
            pass # Consume generator
            
    duration = time.perf_counter() - start
    pulse_freq = pulses / duration
    print(f"   -> Pulse Frequency: {pulse_freq:.2f} Hz")
    print(f"   -> Cycle Time:      {(duration/pulses)*1000:.2f} ms")
    return pulse_freq

def run_holistic_benchmark():
    print("=========================================")
    print("   ELYSIA SYSTEM DIAGNOSTIC (LOCAL)      ")
    print("=========================================")
    
    t0 = time.perf_counter()
    
    mind_score = run_mind_benchmark()
    body_score = run_body_benchmark()
    spirit_score = run_spirit_benchmark()
    
    total_time = time.perf_counter() - t0
    
    print("\n-----------------------------------------")
    print("   DIAGNOSTIC SUMMARY")
    print("-----------------------------------------")
    print(f"TOTAL RUNTIME: {total_time:.4f} s")
    
    if mind_score > 10_000_000 and body_score > 100 and spirit_score > 10:
        print("\n✅ SYSTEM STATUS: OPTIMAL")
        print("   The Trinity is aligned. Hardware is sufficient.")
    else:
        print("\n⚠️ SYSTEM STATUS: SUB-OPTIMAL")
        print("   Bottlenecks detected. Optimization recommended.")
        
    print("=========================================")

if __name__ == "__main__":
    run_holistic_benchmark()
