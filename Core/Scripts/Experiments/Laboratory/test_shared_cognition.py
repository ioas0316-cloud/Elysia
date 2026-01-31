"""
Test Shared Cognition (Phase 18.3 Verification)
===============================================
Core.1_Body.L1_Foundation.Foundation.Network.test_shared_cognition

Verifies that a Main Node can dispatch a 'Thought Packet' 
and a Satellite Node can receive and process it.
"""

import time
import logging
import threading
from Core.1_Body.L1_Foundation.Foundation.Network.aura_pulse import AuraPulse

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger("TestCognition")

def mock_satellite_logic(satellite):
    """Simulates the Satellite Node's behavior loop."""
    def handle_packet(payload, sender):
        task_type = payload.get("type")
        data = payload.get("data")
        logger.info(f"🛰️ [SAT] Processing Task: {task_type} -> {data}")
        
        if task_type == "TEXT_PROCESS":
            res = data[::-1].upper()
            logger.info(f"🛰️ [SAT] Result: {res}")

    satellite._handle_task = handle_packet
    satellite.start_listening()
    
    # Keep alive for test duration
    time.sleep(5)
    satellite.stop()

def run_test():
    logger.info("🧪 Starting Shared Cognition Test...")
    
    # 1. Start Satellite
    sat = AuraPulse(node_type="SATELLITE")
    sat_thread = threading.Thread(target=mock_satellite_logic, args=(sat,))
    sat_thread.start()
    
    # 2. Start Main Node
    main = AuraPulse(node_type="MAIN")
    main.start_pulse()
    
    logger.info("⏳ Waiting for Resonance (Discovery)...")
    time.sleep(2) # Allow handshake
    
    # 3. Dispatch Task
    if sat.node_id in main.peers:
        logger.info(f"✅ Peer Discovered: {sat.node_id}")
        
        task_payload = {"type": "TEXT_PROCESS", "data": "Elysia"}
        logger.info(f"🧠 [MAIN] Dispatching Thought Packet: {task_payload}")
        
        success = main.dispatch_task(sat.node_id, task_payload)
        if success:
            logger.info("🚀 [MAIN] Task Dispatched Successfully.")
        else:
            logger.error("❌ [MAIN] Dispatch Failed.")
    else:
        logger.error("❌ Peer Discovery Failed.")
        
    # Cleanup
    time.sleep(1)
    main.stop()
    sat_thread.join()
    logger.info("✅ Test Complete.")

if __name__ == "__main__":
    run_test()
