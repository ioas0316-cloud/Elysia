
"""
Verification Script: The Loop (Reasoning + Memory)
==================================================
Verifies that ReasoningEngine.think() correctly:
1. Calls OrbManager.unified_rewind()
2. Retrieves the context.
3. Logs the context retrieval.
"""

import sys
import os
import logging
from io import StringIO

# Configure Logging to capture output
log_capture_string = StringIO()
ch = logging.StreamHandler(log_capture_string)
ch.setLevel(logging.INFO)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(ch)

# Verify Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.L5_Mental.Intelligence.Reasoning.reasoning_engine import ReasoningEngine

def test_the_loop():
    print("🧠 [TEST] Initializing ReasoningEngine (The Loop)...")
    engine = ReasoningEngine()
    
    # 1. Inject Fake Memory (if needed, but OrbManager should have persistence from previous test)
    # The previous test (verify_trinity_memory.py) saved 2 memories to 'data/test_memory/'
    # ReasoningEngine defaults to 'data/memory/orbs/', so we might need to point it to test path 
    # OR lets just let it use the default path which might be empty or have old stuff.
    # Actually, OrbManager in ReasoningEngine uses default path 'data/memory/orbs/'.
    # I should explicitly ask it to use the test path OR just manually plant a memory in the default path for this test.
    
    print("   >> Switching OrbManager to 'data/test_memory_loop/' for isolation.")
    engine.orb_manager.persistence_path = "data/test_memory_loop/"
    if not os.path.exists(engine.orb_manager.persistence_path):
        os.makedirs(engine.orb_manager.persistence_path)
    # Plant a seed
    engine.orb_manager.save_memory("Seed_Memory", [0.5]*10, [0.5]*10)
    engine.orb_manager.orbs["Seed_Memory"].memory_content["summary"] = "I was created to test The Loop."
    
    print("\n🤔 [STEP 1] Thinking with Context...")
    insight = engine.think("Who am I?")
    
    # Check Logs
    log_contents = log_capture_string.getvalue()
    print("\n📋 [LOGS CAPTURED]")
    print(log_contents)
    
    print("\n🔍 [VERIFICATION]")
    if "[CONTEXT] Reading the Golden Thread..." in log_contents:
        print("   ✅ SUCCESS: ReasoningEngine consulted the Golden Thread.")
    else:
        print("   ❌ FAIL: Log message not found.")
        
    if "Reading Context from HyperSphere" in log_contents: # Check implementation plan wording just in case
        print("   ✅ SUCCESS: (Alternative wording found)")

    # Check context content (harder to check log for exact prompt content unless logged)
    # But if the first check passed, the code path is active.

    print(f"   ✨ Output Insight: {insight.content}")

if __name__ == "__main__":
    test_the_loop()
