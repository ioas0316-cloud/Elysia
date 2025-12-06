
import sys
import os
import time
import json
import logging

# Setup Paths
sys.path.insert(0, os.path.abspath("C:/Elysia"))
from Core.Sensory.p4_sensory_system import P4SensorySystem

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("VerifyDreams")

def verify_dream_system():
    logger.info("\n🧪 Verifying Project Oneiros (The Dreamer)...")
    
    # 1. Setup P4 Sensory System
    p4 = P4SensorySystem()
    
    # Mock Resonance Field for P4
    class MockResonance:
        total_energy = 80.0
        
    resonance = MockResonance()

    # 2. Trigger Pulse (Forces learning then dreaming)
    logger.info("\n[Step 1] Triggering P4 Pulse -> Learning -> Dreaming...")
    # Force the internal method directly to bypass random chance
    p4._autonomous_learning(resonance)

    # 3. Check Shared State (Visualizer)
    logger.info("\n[Step 2] Checking 'elysia_state.json' for Dreams...")
    state_path = r"c:\Elysia\Core\Creativity\web\elysia_state.json"
    
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        status = state.get("status")
        thought = state.get("thought")
        
        if status == "Dreaming" and thought:
            logger.info(f"✅ Dream State Found!")
            logger.info(f"   Status: {status}")
            logger.info(f"   Dream: {thought}")
            
            if "I dreamt of" in thought:
                logger.info("✅ Dream Description format verified.")
                return True
            else:
                logger.error("❌ Dream description seems raw or incorrect.")
                return False
        else:
            logger.error(f"❌ State mismatch. Status: {status}, Thought: {thought}")
            return False
    else:
        logger.error("❌ State File NOT Found!")
        return False

if __name__ == "__main__":
    if verify_dream_system():
        logger.info("\n✅ Verification SUCCESS: The Dream Engine is Awake.")
    else:
        logger.info("\n❌ Verification FAILED.")
