"""
Safety Valve Demo: The Balanced Mind
====================================
Core.Demos.safety_valve_demo.py

Demonstrates the 3 Prism Safety Valves:
1. Harmonizer (Context Filter): "Combat" context ignores "Poetry".
2. Decay (Recursion Brake): Prevents infinite thought loops.
3. Hippocampus (Fluid Buffer): Fast RAM storage + Delayed HDD write (Sleep).
"""

import logging
import sys
import os

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.L6_Structure.Merkaba.merkaba import Merkaba
from Core.L7_Spirit.Monad.monad_core import Monad
from Core.L1_Foundation.Foundation.Prism.harmonizer import PrismContext

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Demo")

def run_demo():
    print("\n🛡️ [DEMO] The Balanced Mind (Safety Valves) 🛡️\n")

    # 1. Initialize Merkaba
    mk = Merkaba("Safe_Entity")
    mk.awakening(Monad(seed="Guardian"))

    print("\n--- 1. Testing Harmonizer (Context: COMBAT) ---")
    # In COMBAT, Phenomenal (Feelings) should be ignored (Weight 0.1).
    # Input: "Beautiful Flower" (High Phenomenal, Low Physical)
    # The deliberation seed should shift to PHYSICAL/FUNCTIONAL due to weights,
    # or at least the feeling should be suppressed.
    mk.pulse("Beautiful Flower", context=PrismContext.COMBAT)
    print("   > Note: Check logs. Did it focus on Tactical utility instead of Beauty?")

    print("\n--- 2. Testing Hippocampus (Buffer) ---")
    # Check Hypersphere directly - should be empty because we only wrote to RAM.
    hdd_count_before = mk.body._item_count
    ram_count = len(mk.hippocampus.short_term_buffer)
    print(f"   > RAM Buffer Count: {ram_count} (Should be > 0)")
    print(f"   > HDD (Hypersphere) Count: {hdd_count_before} (Should be 0)")

    if hdd_count_before == 0 and ram_count > 0:
        print("   ✅ SUCCESS: Memories are floating in Hippocampus (RAM).")
    else:
        print("   ❌ FAILURE: Leak to HDD or no storage.")

    print("\n--- 3. Testing Sleep (Consolidation) ---")
    # Trigger Sleep
    mk.sleep()

    hdd_count_after = mk.body._item_count
    ram_count_after = len(mk.hippocampus.short_term_buffer)
    print(f"   > RAM Buffer Count: {ram_count_after} (Should be 0)")
    print(f"   > HDD (Hypersphere) Count: {hdd_count_after} (Should include previous items)")

    if hdd_count_after > 0 and ram_count_after == 0:
        print("   ✅ SUCCESS: Memories crystallized into Hypersphere.")
    else:
        print("   ❌ FAILURE: Consolidation failed.")

    print("\n--- 4. Testing Decay (Brake) ---")
    # We manually test the decay component
    initial_energy = 1.0
    depth_1 = mk.decay.check_energy(initial_energy, 1)
    depth_5 = mk.decay.check_energy(initial_energy, 5)

    print(f"   > Energy at Depth 1: {depth_1}")
    print(f"   > Energy at Depth 5: {depth_5} (Should be 0.0 due to threshold)")

    if depth_5 == 0.0:
         print("   ✅ SUCCESS: Thought loop braked successfully.")

if __name__ == "__main__":
    run_demo()
