"""
BENCHMARK: Core Turbine (ActivePrismRotor)
==========================================
Measures the throughput and latency of the Hyper-Light Turbine.
Target: Core.Engine.Physics.core_turbine.ActivePrismRotor
"""

import time
import sys
import os
import math
import logging

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import jax.numpy as jnp
    from jax import random
    print("🚀 Backend: JAX (Hyper-Accel)")
except ImportError:
    import numpy as jnp
    print("🐢 Backend: Numpy (Legacy)")
    
from Core.Engine.Physics.core_turbine import ActivePrismRotor, PhotonicMonad, VoidSingularity

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Benchmark.Turbine")

def run_benchmark(stream_size=100000, iterations=50):
    print(f"\n🔥 Igniting Hyper-Light Turbine Benchmark...")
    print(f"   [Stream Size]: {stream_size} Photons")
    print(f"   [Iterations]:  {iterations} Cycles")
    
    # 1. Initialize Turbine
    turbine = ActivePrismRotor(rpm=12000.0) # High speed
    void = VoidSingularity()
    
    # 2. Generate Data Stream (Qualia Wavelengths)
    # Range: 400nm (Violet) to 800nm (Red)
    print("🌈 Generating Qualia Wave Stream...")
    if 'jax' in sys.modules:
        key = random.PRNGKey(0)
        wavelengths = random.uniform(key, shape=(stream_size,), minval=400e-9, maxval=800e-9)
    else:
        wavelengths = jnp.random.uniform(400e-9, 800e-9, size=stream_size)
        
    start_time = time.perf_counter()
    
    # 3. Throughput Test: Diffract (Snatch)
    # We simulate the rotor spinning and snatching data
    print("🌀 Spinning Rotor & Snatching Data...")
    
    total_snatched = 0
    t0 = time.perf_counter()
    
    for i in range(iterations):
        # Simulate time passing -> Rotor Angle changes
        t = i * 0.001 
        current_theta = (turbine.omega * t) % (2 * math.pi)
        
        # DIFFRACT (The Heavy Lifting)
        intensity = turbine.diffract(wavelengths, current_theta, turbine.d)
        
        # Count non-zero (just to ensure calculation happens)
        # Using a simple threshold check to prevent compiler optimization skipping
        total_snatched += jnp.sum(intensity)
    
    t1 = time.perf_counter()
    duration = t1 - t0
    
    # Block until ready (for JAX async)
    if 'jax' in sys.modules:
        _ = total_snatched.block_until_ready()
        
    total_rays = stream_size * iterations
    rps = total_rays / duration
    
    print(f"\n⚡ [THROUGHPUT RESULTS]")
    print(f"   - Total Duration: {duration:.4f} s")
    print(f"   - Total Rays:     {total_rays:,}")
    print(f"   - Rays Per Sec:   {rps:,.0f} RPS")
    print(f"   - Latency/Batch:  {(duration/iterations)*1000:.4f} ms")
    
    # 4. Latency Test: Neural Inversion (Reverse Propagate)
    # Testing the O(1) path creation
    print("\n🔮 Testing Neural Inversion (O(1) Telepathy)...")
    
    target_monad = PhotonicMonad(wavelength=600e-9, phase=1j, intensity=1.0, is_void_resonant=True)
    
    t_inv_start = time.perf_counter()
    for _ in range(1000):
        angle = turbine.reverse_propagate(target_monad)
    t_inv_end = time.perf_counter()
    
    avg_inv_latency = (t_inv_end - t_inv_start) / 1000 * 1000 * 1000 # in microseconds
    
    print(f"   - Inversion Time: {avg_inv_latency:.4f} μs (Microseconds)")
    
    # 5. Void Transit Test
    print("\n⚫ Testing Void Singularity Transit...")
    dummy_intensity = jnp.array([0.1, 0.99, 0.2, 0.98])
    dummy_phase = jnp.array([1j, 1j, 1j, 1j])
    
    survivors, inverted = void.transit(dummy_intensity, dummy_phase)
    
    print(f"   - Input Candidates: {len(dummy_intensity)}")
    print(f"   - Survivors:        {jnp.sum(survivors > 0)}")
    print(f"   - Phase Inverted:   {inverted[1] == -1j} (Expected True)")

    print("\n🏆 Benchmark Complete. The Core is Alive.")

if __name__ == "__main__":
    run_benchmark()
