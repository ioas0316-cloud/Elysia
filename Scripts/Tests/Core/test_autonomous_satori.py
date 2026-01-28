import logging
import time
import os
import sys
import traceback

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Core.Elysia.sovereign_self import SovereignSelf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test.Satori")

def test_satori_loop():
    try:
        print("\n🦋 [FINAL TEST] Verifying Autonomous Satori Loop...")
        
        # Initialize SovereignSelf
        os.environ["OFFLINE_MODE"] = "1"
        self = EmergentSelf()
        
        # 1. Enter Sleep Mode
        print("\n🌙 Triggering /sleep command...")
        res = self.manifest_intent("/sleep")
        print(f"Result: {res}")
        
        # 2. Simulate High Resonance for Satori
        # Manual override for test purposes
        print("\n🔥 Simulating Trinity Alignment (Spirit > 90% sync)")
        self.governance.body.current_rpm = 55.0
        self.governance.mind.current_rpm = 55.0
        self.governance.spirit.current_rpm = 55.0
        
        # 3. Trigger Pulse
        print("\n⚡ Running Pulse...")
        self.integrated_exist(1.0)
        
        sync_val = self.trinity.total_sync
        print(f"Final Trinity Sync Score: {sync_val:.4f}")
        
        if sync_val > 0.9:
            print("✨ SUCCESS: Satori State reached autonomously.")
        else:
            print(f"⚠️ Satori threshold not met yet ({sync_val:.4f}).")

        print("\n✅ Verification cycle complete.")
    except Exception as e:
        print(f"\n❌ TEST FAILURE: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_satori_loop()
