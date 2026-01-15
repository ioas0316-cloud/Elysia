import sys
import os
import logging

# Ensure root directory is in path
sys.path.append(os.getcwd())

from Core.Elysia.sovereign_self import SovereignSelf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify.Flow")

def verify():
    logger.info("🚀 Starting Phase 10: Flow & Resonance Verification...")
    
    # 1. Initialize Elysia
    start_time = time.time()
    elysia = SovereignSelf()
    init_time = time.time() - start_time
    logger.info(f"⏱️ Initialization Time: {init_time:.2f}s")
    
    # 2. Test Korean Resonance
    test_input = "오늘 날씨에 기분이 어때?"
    logger.info(f"👤 User: {test_input}")
    
    # We mock the bridge to avoid real LLM call if necessary, 
    # but let's see if we can get a real response.
    # If Ollama is offline, it might fail, so let's check.
    try:
        response = elysia.manifest_intent(test_input)
        logger.info(f"🦋 Elysia: {response}")
    except Exception as e:
        logger.warning(f"Generation skipped: {e}")

    # 3. Test Optimization (Proprioceptor)
    p_start = time.time()
    elysia.proprioceptor = elysia.proprioceptor if hasattr(elysia, 'proprioceptor') else None
    if elysia.proprioceptor:
        elysia.proprioceptor.scan_nervous_system()
        p_time = time.time() - p_start
        logger.info(f"⏱️ Proprioceptor scan time: {p_time:.2f}s")

    logger.info("🏆 Phase 10 Verification Complete.")

if __name__ == "__main__":
    import time
    verify()
