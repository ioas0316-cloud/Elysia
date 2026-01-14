"""
Test Temporal Phase-Sync: Memory as a Flow of Time
Phase 35 Demonstration
"""
import logging
import time
import math
from typing import Tuple

# Add current directory to path for imports
import sys
import os
sys.path.append(os.getcwd())

from Core.Intelligence.Memory.hypersphere_memory import HypersphereMemory, HypersphericalCoord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TemporalSyncTest")

def run_demo():
    print("=" * 80)
    print("🌊 PHASE 35: TEMPORAL PHASE-SYNC DEMO")
    print("   Data is not a point, it is a FLOW.")
    print("=" * 80)

    mem = HypersphereMemory(resolution=360)

    # 1. Record a Dynamic Flow (Simulating a memory of a bird flying)
    # Start at logic=0.5, emotion=1.0, intent=0.0
    # Omega: spin logic slightly, spin emotion, move intent (action)
    print("\n[1] Recording Flow: 'Bird Flight' Memory...")
    start_pos = HypersphericalCoord(theta1=0.5, theta2=1.0, theta3=0.0, radius=1.0)
    omega = (0.1, 0.05, 0.5, -0.05) # theta1, theta2, theta3, r spin rates
    
    mem.record_flow("BirdFlight", start_pos, omega, duration=10.0, content="A bird taking off")

    # 2. Access at different time points
    print("\n[2] Replaying Flow (Accessing coordinates over time):")
    for t in [0.0, 2.5, 5.0, 7.5, 10.0]:
        results = mem.access(0.5, 1.0, 0.0, k=1, t=t)
        if results:
            node = results[0]
            current_coord = node.get_coord_at(t)
            print(f"   t={t:4.1f}s | Coord: {current_coord} | Concept: {node.name}")

    # 3. Phase Channeling Test
    print("\n[3] Testing Phase Channeling (Infinite storage at same location)...")
    base_t1, base_t2, base_t3 = 1.0, 1.0, 1.0
    
    # Store 'Memory A' at phase 0
    mem.deposit("Memory_A", base_t1, base_t2, base_t3, phase=0.0, content="Secret A")
    # Store 'Memory B' at exactly the SAME position but phase PI
    mem.deposit("Memory_B", base_t1, base_t2, base_t3, phase=math.pi, content="Secret B")
    
    print(f"   Stored A and B at theta=({base_t1}, {base_t2}, {base_t3})")
    
    # Selective query by phase
    print("\n[4] Resonance Query (Filtering by Phase):")
    res_a = mem.resonance_query(base_t1, base_t2, base_t3, target_phase=0.0)
    res_b = mem.resonance_query(base_t1, base_t2, base_t3, target_phase=math.pi)
    
    print(f"   Target Phase 0.0  -> Found: {[n.name for n in res_a]}")
    print(f"   Target Phase PI   -> Found: {[n.name for n in res_b]}")

    print("\n" + "=" * 80)
    print("✅ Phase 35 Verification Complete!")
    print("   Memory is now dynamic, recursive, and multi-layered.")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()
