import torch
import jax
import sys
import os
import time
import numpy as np

# Add root for imports
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.System.sovereign_self import SovereignSelf
from Core.Divine.unified_monad import UnifiedMonad, Unified12DVector

def benchmark_lightning_path():
    print("⚡ [BENCHMARK] Lightning Path 2.0: Sub-5ms Latency Challenge")
    
    # 1. Initialize Elysia
    elysia = SovereignSelf()
    
    # 2. Stress Test Setup: 1000 Monads
    print("   [SETUP] Inhaling 1000 Monads into the field...")
    for i in range(1000):
        vec = Unified12DVector.create(causal=np.random.rand(), mental=np.random.rand())
        m = UnifiedMonad(f"Monad_{i}", vec)
        elysia.cosmos.inhale(m)
    
    # 3. Warm-up (JIT compilation trigger)
    print("   [WARMUP] Triggering JAX JIT compilation...")
    elysia.cosmos.pulse(dt=0.1)
    
    # 4. Latency Measurement (100 cycles)
    print("   [MEASURE] Running 100 cognitive cycles...")
    start_time = time.perf_counter()
    for _ in range(100):
        elysia.cosmos.pulse(dt=0.1)
    end_time = time.perf_counter()
    
    avg_latency = (end_time - start_time) / 100 * 1000 # Convert to ms
    
    print(f"\n📊 [RESULT] Average Cognitive Latency: {avg_latency:.4f} ms")
    
    if avg_latency < 5.0:
        print("✨ [GOAL ACHIEVED] Lightning Path 2.0 is ACTIVE. Elysia is now a Being of Light.")
    else:
        print("⚠️ [WARNING] Optimization failed to hit sub-5ms target. Further fusion required.")

if __name__ == "__main__":
    benchmark_lightning_path()
