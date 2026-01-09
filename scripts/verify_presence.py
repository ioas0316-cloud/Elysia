
import sys
import os
import time
import logging

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.Autonomy.elysian_heartbeat import ElysianHeartbeat

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

def verify_presence():
    print("🚀 [VERIFICATION] Testing the Living Presence...")
    
    heart = ElysianHeartbeat()
    
    # 1. Trigger Genesis to update 'latest_creation'
    heart.soul_mesh.variables["Inspiration"].value = 0.8
    
    # 2. Run one cycle of the HEARTBEAT manifest logic
    print("\n💓 Pumping the Heartbeat...")
    heart.is_alive = True
    
    # Simulate the loop logic manually for verification
    # Genesis
    heart.latest_creation = heart.genesis.manifest(0.8)
    
    # Update Presence
    heart._refresh_presence()
    
    # 3. Check if ELYSIA_PRESENCE.md exists and is clean
    presence_path = "c:/Elysia/ELYSIA_PRESENCE.md"
    if os.path.exists(presence_path):
        print(f"\n✅ ELYSIA_PRESENCE.md created at root.")
        with open(presence_path, "r", encoding="utf-8") as f:
            print("\n--- [ PRESENCE FILE CONTENT ] ---")
            print(f.read())
            print("--- [ END OF CONTENT ] ---\n")
    else:
        print("❌ Presence file NOT found!")

    # 4. Check if brain is clean
    with open("c:/Elysia/Core/Memory/semantic_field.json", "r", encoding="utf-8") as f:
        import json
        data = json.load(f)
        leaks = [c['meaning'] for c in data['concepts'] if "I feel deeply" in c['meaning']]
        if not leaks:
            print("✅ Brain is clean. No prompt leaks detected.")
        else:
            print(f"⚠️ Found {len(leaks)} residual leaks in the brain.")

if __name__ == "__main__":
    verify_presence()
