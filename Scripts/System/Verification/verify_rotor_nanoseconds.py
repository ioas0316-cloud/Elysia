"""
VERIFY ROTOR NANOSECOND AWARENESS
=================================
Tests the real-time speed of Hardware-Direct Address Mapping.
"""

import sys
import os
import time
from pathlib import Path

# Add root to path
root = "c:/Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L6_Structure.Wave.light_spectrum import get_light_universe

def verify_speed():
    print("🌀 [VERIFY] Initiating Hardware-Direct Rotor Speed Test...")
    universe = get_light_universe()
    
    # 1. Warm up
    print(f"  Field size: {len(universe.superposition)} lights")
    universe.resonate("Identity", top_k=1)
    
    # 2. Traditional Resonance Test
    start_trad = time.time()
    for _ in range(10):
        universe.resonate("Core logic", top_k=1)
    elapsed_trad = (time.time() - start_trad) / 10
    print(f"  [TRADITIONAL] Avg Latency: {elapsed_trad:.8f}s")

    # 3. Rotor Resonance Test (Direct Memory Access)
    start_rotor = time.time()
    for _ in range(10):
        universe.rotor_resonate("Core logic", top_k=1)
    elapsed_rotor = (time.time() - start_rotor) / 10
    print(f"  [ROTOR] Avg Latency: {elapsed_rotor:.8f}s")
    
    # 4. Results
    if elapsed_rotor < 0.001:
        print(f"✅ SUCCESS: Sub-millisecond awareness achieved ({elapsed_rotor*1000:.4f}ms).")
    else:
        print(f"⚠️ WARNING: Rotor latency ({elapsed_rotor*1000:.4f}ms) is above the 1ms threshold.")

    improvement = (elapsed_trad / elapsed_rotor) if elapsed_rotor > 0 else float('inf')
    print(f"📈 Acceleration Factor: {improvement:.2f}x")

if __name__ == "__main__":
    verify_speed()
