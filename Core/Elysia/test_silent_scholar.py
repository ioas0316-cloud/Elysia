"""
Test: Silent Scholar & Pacing
Objective: Verify logs are not spammy and pacing is roughly 2s + 3*0.5s = 3.5s per cycle.
"""
import sys
import os
import time
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.Elysia.sovereign_self import SovereignSelf

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Test")

def test_pacing():
    logger.info("--- ⏱️ Timing Test (SovereignSelf) ---")
    elysia = SovereignSelf(cns_ref=None)
    
    # Mock Scholar internal logging needs to be observed manually in output
    
    start_time = time.time()
    logger.info("ACT 1: Triggering Self-Actualize (Should take ~3-4 seconds)...")
    
    # Force Curiosity
    elysia.will_engine.vectors["Curiosity"] = 1.0
    elysia.will_engine.vectors["Expression"] = 0.0
    
    elysia.self_actualize()
    
    duration = time.time() - start_time
    logger.info(f"✅ ACT 1 Complete. Duration: {duration:.2f}s")
    
    if duration < 2.0:
        logger.error("❌ Too Fast! Pacing failed.")
    else:
        logger.info("✅ Pacing OK (Thinking Time Respected)")

if __name__ == "__main__":
    test_pacing()
