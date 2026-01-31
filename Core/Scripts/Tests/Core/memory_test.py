"""
TEST: MEMORY CONSOLIDATION
==========================
Verifies Phase 19 Soul Architecture.
"""
import sys
import os
import json
import logging
from datetime import datetime

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.L7_Spirit.Soul.logbook import Logbook
from Core.L7_Spirit.Soul.growth_graph import GrowthTracker

logging.basicConfig(level=logging.INFO)

def run_test():
    print("==================================")
    print("   PHASE 19: MEMORY TEST          ")
    print("==================================")

    test_dir = "c:/Elysia/data/Logs/MemoryTest"
    mem_dir = "c:/Elysia/data/Memories/Chronicles/Test"
    
    os.makedirs(test_dir, exist_ok=True)
    
    # 1. Create Dummy Logs
    print("\n👉 [SETUP] Creating dummy logs...")
    log_file = os.path.join(test_dir, "action_history.jsonl")
    
    entries = [
        {"timestamp": "2026-01-20T10:00:00", "phase": "EXECUTION", "intent": "Test 1", "action_type": "TEST"},
        {"timestamp": "2026-01-20T10:00:01", "phase": "REFLECTION", "result_status": "SUCCESS", "score": 0.9, "result_data": "Great success"},
        {"timestamp": "2026-01-20T11:00:00", "phase": "EXECUTION", "intent": "Test 2", "action_type": "TEST"},
        {"timestamp": "2026-01-20T11:00:01", "phase": "REFLECTION", "result_status": "ERROR", "score": -0.5, "result_data": "Big failure"}
    ]
    
    with open(log_file, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    # 2. Test Chronicler
    print("\n👉 [CHRONICLER] Consolidating Memory...")
    logbook = Logbook(log_dir=test_dir, memory_dir=mem_dir)
    chronicle_path = logbook.consolidate_memory("2026-01-20")
    
    if os.path.exists(chronicle_path):
        print(f"   -> ✅ Chronicle created: {chronicle_path}")
        with open(chronicle_path, "r", encoding="utf-8") as f:
            print(f"   -- PREVIEW --\n{f.read()[:200]}...")
    else:
        print("   -> ❌ Chronicle Failed.")

    # 3. Test Tracker
    print("\n👉 [TRACKER] Updating Growth Graph...")
    tracker = GrowthTracker(history_file="c:/Elysia/data/Memories/growth_stats_test.csv")
    
    # Mock stats
    stats = {"total_actions": 2, "avg_resonance": 0.2, "max_resonance": 0.9}
    tracker.update_growth_stats("2026-01-20", stats)
    
    if os.path.exists(tracker.history_file):
         print(f"   -> ✅ Graph Data Updated: {tracker.history_file}")
    else:
         print("   -> ❌ Graph Failed.")

    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    run_test()
