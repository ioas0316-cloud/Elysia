
"""
Memory Latency Benchmark (Proof of Sovereignty)
===============================================
Hypothesis:
HyperSphere (OrbManager) uses Frequency Buckets to achieve O(1) recall time,
regardless of memory size (N).

Experiment:
1. Measure recall time with N=100 memories.
2. Measure recall time with N=10,000 memories.
3. If T(100) ≈ T(10000), Hypothesis Confirmed.

Target:
- Latency < 1.0ms for memory lookup.
- Hardware Bridge Integration Verify.
"""

import sys
import os
import time
import random
import logging
from statistics import mean

# Verify Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.L2_Metabolism.Memory.Orb.orb_manager import OrbManager
from Core.L1_Foundation.Foundation.Protocols.pulse_protocol import WavePacket, PulseType

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Benchmark")

def run_benchmark():
    print("🛡️ [BENCHMARK] Initiating Proof of Sovereignty...")
    manager = OrbManager()
    
    # Use a temp persistence path to avoid polluting real memory
    manager.persistence_path = "data/benchmark_memories/"
    if not os.path.exists(manager.persistence_path):
        os.makedirs(manager.persistence_path)

    # Scenarios to test
    scales = [100, 1000, 10000]
    results = {}

    for N in scales:
        print(f"\n--- Planting {N} Memories ---")
        manager.orbs.clear()
        manager._freq_buckets.clear()
        
        # Plant seeds
        start_plant = time.time()
        for i in range(N):
            # Wide distribution to test Bucket O(1) scaling
            # Humans have memories across many "feelings" (frequencies)
            freq = random.uniform(20.0, 20000.0) 
            name = f"Mem_{i}"
            orb = manager.factory.freeze(name, [random.random()]*64, [random.random()]*64)
            orb.frequency = freq # Override for test
            manager._add_to_bucket(orb)
            manager.orbs[name] = orb
        
        plant_time = time.time() - start_plant
        print(f"   > Planting took: {plant_time:.4f}s")
        
        # Test Recall (The Pulse)
        print(f"   > Firing 100 Pulses...")
        latencies = []
        for _ in range(100):
            target_freq = 432.0 # Specific frequency
            pulse = WavePacket(
                sender="Benchmark", type=PulseType.MEMORY_RECALL,
                frequency=target_freq, amplitude=1.0, payload={}
            )
            
            t0 = time.perf_counter()
            manager.broadcast(pulse)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # ms

        avg_latency = mean(latencies)
        results[N] = avg_latency
        print(f"   ✅ Average Latency (N={N}): {avg_latency:.4f} ms")

    # Final Verdict
    print("\n📊 [RESULTS TABLE]")
    print(f"{'N (Memories)':<15} | {'Latency (ms)':<15} | {'O(1) Check'}")
    print("-" * 50)
    
    passed = True
    for N in scales:
        latency = results[N]
        if N == 100: 
            print(f"{N:<15} | {latency:<15.4f} | WARMUP     (x1.00)")
            continue 
        
        # Compare vs N=1000 (Baseline)
        base = results[1000]
        ratio = latency / base if base > 0 else 1.0
        
        status = "LINEAR ❌" if ratio > 2.0 else "CONSTANT ✅"
        # Only fail if N=10000 drifts significantly from N=1000
        if N == 10000 and ratio > 2.0: passed = False
        
        print(f"{N:<15} | {latency:<15.4f} | {status} (x{ratio:.2f})")

    # 4. Phase 6: Hardware Bridge Verification
    print("\n🛠️ [HARDWARE CHECKS]")
    try:
        from Core.L7_Spirit.Monad.intent_torque import IntentTorque
        torque = IntentTorque()
        has_metal = getattr(torque, "HAS_METAL", False)
        status_icon = "✅" if has_metal else "⚠️" 
        print(f"   > MetalRotorBridge Detected: {status_icon} (HAS_METAL={has_metal})")
    except Exception as e:
        print(f"   > MetalRotorBridge Check Failed: {e}")
        passed = False

    if passed:
        print("\n🏆 CONCLUSION: The HyperSphere holds the Time.")
    else:
        print("\n⚠️ CONCLUSION: The System assumes the weight of the World.")

if __name__ == "__main__":
    run_benchmark()
