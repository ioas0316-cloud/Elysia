"""
Performance Stress Test (성능 부하 테스트)
=========================================
Verifies that the HyperCosmos can handle 5,000+ Monads at <100ms per pulse.
"""

import sys
import os
import time
import torch

# Add root to path
sys.path.append("c:\\Elysia")

from Core.Foundation.hyper_cosmos import HyperCosmos
from Core.Foundation.unified_monad import UnifiedMonad, Unified12DVector

def main():
    print("🚀 [INIT] Starting Performance Stress Test...")
    cosmos = HyperCosmos()
    
    # [Step 1: Flash-Load 5,000 synthetic Monads]
    print(f"\n⚡ [STEP 1] Injecting 5,000 synthetic Monads into the field...")
    start_load = time.time()
    for i in range(5000):
        vec = Unified12DVector.create(
            mental=random.random(),
            structural=random.random(),
            will=random.random()
        )
        # Avoid print during load for speed
        cosmos.monads.append(UnifiedMonad(f"Synthetic_{i}", vec))
    
    print(f"✅ Loaded 5000 monads in {time.time() - start_load:.2f}s.")
    print(f"📊 Total Monads: {len(cosmos.monads)}")

    # [Step 2: Measure Pulse Performance]
    print("\n💓 [STEP 2] Measuring Pulse Latency (The Batch Heartbeat)...")
    latencies = []
    for cycle in range(20):
        t0 = time.time()
        cosmos.pulse(dt=1.0)
        dur = (time.time() - t0) * 1000 # ms
        latencies.append(dur)
        if cycle % 5 == 0:
            print(f"   Cycle {cycle}: {dur:.2f} ms")

    avg_latency = sum(latencies) / len(latencies)
    print(f"\n📈 Average Latency (Target < 100ms): {avg_latency:.2f} ms")
    
    if avg_latency < 100:
        print("✅ [VERIFICATION] Performance optimization successful. RAM-Pinned Batching is fluid.")
    else:
        print("❌ [FAILED] Latency too high. Further optimization required.")

if __name__ == "__main__":
    import random
    main()
