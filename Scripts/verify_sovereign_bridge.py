import logging
import sys
import os

# Ensure root is in path
root = "c:/Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.L6_Structure.M5_Engine.sovereign_bridge import bridge

def test_bridge():
    print("🌉 [BRIDGE_TEST] Scanning for life...")
    # Force a scan (bridge does this on init, but let's be sure for the new file)
    bridge.nerve.scan_body()
    
    print(f"   -> Registry contains {len(bridge.nerve.organ_map)} units.")
    
    # 1. Locate the test heart
    heart_cls = bridge.get_cell("TestHeart")
    if heart_cls:
        print("✅ Found TestHeart!")
        h = heart_cls()
        h.beat()
    else:
        print("❌ Failed to find TestHeart.")
        print("Existing keys:", list(bridge.nerve.organ_map.keys()))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_bridge()
