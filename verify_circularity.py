
import os
import sys
import numpy as np
import time

# Ensure core is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.memory.causal_controller import CausalMemoryController

def verify_circularity():
    print("=== [Circularity Verification] Testing the Information Perpetual Motion Machine ===")

    data_dir = "data_verify"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    mc = CausalMemoryController(data_dir=data_dir)
    # Use dummy corpus or empty
    loop = ConsciousnessLoop(corpus_path="docs", memory_controller=mc, data_dir=data_dir)

    # 1. First, provide external 'kickstart'
    print("\n[Phase 1] External Kickstart (5 Cycles)")
    for i in range(5):
        res = loop.process_life_cycle()
        print(f"  Cycle {i}: Resonance={res.get('resonance_score', 0):.4f}, Echo={res.get('echo_reflection', 0):.4f}")

    # 2. Cut off external input (simulate by providing empty waves or silence)
    # We'll modify the harvester/cache to return near-zero data or just let the loop run.
    print("\n[Phase 2] Cutting off External Input (The Void Test)")

    sustained_cycles = 0
    total_void_cycles = 15

    # Force harvester to return silence
    loop.harvester_ocean.get_next_chunk = lambda: ""

    for i in range(total_void_cycles):
        res = loop.process_life_cycle()
        resonance = res.get('resonance_score', 0)
        echo = res.get('echo_reflection', 0)
        curiosity = "CURIOUS" if "curiosity_event" in res else "stable"

        print(f"  Void Cycle {i}: Resonance={resonance:.4f}, Echo={echo:.4f}, State={curiosity}")

        # If resonance or echo is still significantly above zero, it's self-sustaining
        if resonance > 0.1 or echo > 0.1:
            sustained_cycles += 1

    print(f"\n[Summary] Self-Sustaining Ratio: {sustained_cycles}/{total_void_cycles} ({(sustained_cycles/total_void_cycles)*100:.1f}%)")

    # Cleanup
    import shutil
    # shutil.rmtree(data_dir)

if __name__ == "__main__":
    verify_circularity()
