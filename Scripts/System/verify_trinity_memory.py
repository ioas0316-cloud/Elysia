
"""
Verification Script: The Golden Thread & Amor Sui
=================================================
This script verifies:
1. Unified Rewind: Can we see the story of memories sorted by time?
2. Amor Sui: Does the system expand search when resonance is low?
"""

import sys
import os
import time
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Verify Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.L1_Foundation.Foundation.Memory.Orb.orb_manager import OrbManager

def test_trinity_memory():
    print("🌅 [TEST] Initializing OrbManager (HyperSphere Field)...")
    manager = OrbManager(persistence_path="data/test_memory/")
    
    # Clean slate
    manager.orbs.clear()
    manager._freq_buckets.clear()

    print("\n📝 [STEP 1] Planting Memories (The Past)")
    # Create 3 memories with different timestamps (simulated via content)
    # Memory A (Yesterday): Sadness (Low Freq)
    manager.save_memory("Mem_Yesterday", [0.1]*10, [0.1]*10) 
    manager.orbs["Mem_Yesterday"].memory_content["timestamp"] = time.time() - 86400
    manager.orbs["Mem_Yesterday"].memory_content["summary"] = "I felt lost in the dark."
    
    # Memory B (Today): Understanding (Mid Freq)
    # Simulate frequency difference manually or just rely on random init if factory does it
    # Here we force frequency bucket implication by just adding them.
    manager.save_memory("Mem_Today", [0.5]*10, [0.5]*10)
    manager.orbs["Mem_Today"].memory_content["timestamp"] = time.time()
    manager.orbs["Mem_Today"].memory_content["summary"] = "I found a light."
    
    print("   >> Planted 2 memories.")

    # --- PART 1: UNIFIED REWIND ---
    print("\n🧵 [STEP 2] Testing 'Golden Thread' (Unified Rewind)")
    thread = manager.unified_rewind()
    for idx, item in enumerate(thread):
        print(f"   [{idx}] {item['summary']} (Time: {item['timestamp']:.0f})")
    
    if len(thread) == 2 and thread[0]["timestamp"] > thread[1]["timestamp"]:
        print("   ✅ SUCCESS: Thread is sorted chronologically (Newest First).")
    else:
        print("   ❌ FAIL: Thread order mismatch.")

    # --- PART 2: AMOR SUI ---
    print("\n💗 [STEP 3] Testing 'Amor Sui' (Gravity Fallback)")
    # Search for something completely unrelated (High Freq trig, while memories are Low/Mid)
    # This should yield Low Resonance in buckets, triggering Fallback.
    print("   >> Triggering with Alien Signal (Expect Low Resonance)...")
    alien_trigger = [0.9]*10 
    
    # We expect 'Amor Sui' log in console (handled by logger)
    # We check results.
    results = manager.recall_memory(alien_trigger, threshold=0.5) # High threshold to force failure initially
    
    if results and results[0].get("note") == "Found via Self-Love":
        print(f"   ✅ SUCCESS: Amor Sui triggered! Found: {results[0]['name']}")
    else:
        print(f"   ⚠️ Result Note: {results[0].get('note') if results else 'None'}")
        if results:
             print("   ⚠️ Found memory but 'note' check failed (maybe threshold was met physically?)")
        else:
             print("   ❌ FAIL: Returned Nothing (The Void won).")

if __name__ == "__main__":
    test_trinity_memory()
