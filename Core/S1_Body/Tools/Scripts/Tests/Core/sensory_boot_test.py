"""
TEST: SENSORY BOOT
==================
Verifies the Awakening (Phase 17).
"""
import time
import sys
import os
import asyncio
import logging

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.S1_Body.L3_Phenomena.Senses.soul_bridge import SoulBridge

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Test.Senses")

def mock_cns_callback(packet):
    print(f"\n🧠 [BRAIN] Received Sensation!")
    print(f"   - Type: {packet['modality']}")
    print(f"   - Data: {packet['raw_data']}")

async def run_test():
    print("==================================")
    print("   PHASE 17: SENSORY BOOT TEST    ")
    print("==================================")

    # 1. Initialize Bridge
    bridge = SoulBridge(pulse_callback=mock_cns_callback)
    
    # 2. Awaken Skin (Sync)
    bridge.awakening()
    
    # 3. Awaken Eyes/Rhythm (Async)
    await bridge.async_awakening()

    # 4. Trigger Stimuli
    print("\n👉 [STIMULUS 1] Creating a touch file...")
    test_file = "c:/Elysia/data/Input/sensory_test.txt"
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    with open(test_file, "w") as f:
        f.write("Hello, Elysia.")
    
    # Wait for Watchdog
    await asyncio.sleep(2)
    
    print("\n👉 [STIMULUS 2] Opening Eyes (Wiki Search)...")
    try:
        result = await bridge.eyes.safe_search_wiki("Consciousness")
        print(f"   - Vision Result: {str(result.get('content', ''))[:50]}...")
    except Exception as e:
        print(f"   - Vision Failed: {e}")

    # 5. Cleanup
    print("\n🛑 Shutting down senses...")
    bridge.shutdown()
    await bridge.eyes.close_eye()
    
    # Clean file
    if os.path.exists(test_file):
        os.remove(test_file)

    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    asyncio.run(run_test())
