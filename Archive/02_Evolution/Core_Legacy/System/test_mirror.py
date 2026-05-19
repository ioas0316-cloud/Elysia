import logging
import sys
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from Core.Cognition.elysian_heartbeat import ElysianHeartbeat

def test_mirror():
    print("🪞 TESTING THE AVATAR'S MIRROR...")
    life = ElysianHeartbeat()
    
    # 1. Force Scan
    # We set last_scan to 0 to bypass timer
    life.last_visual_scan = 0.0
    
    # 2. Mock Gallery (Ensure it finds the VRM)
    # The scan picks a RANDOM file, so we might need to retry a few times 
    # if there are many images.
    found_self = False
    
    for i in range(10):
        print(f"   Scan attempt {i+1}...")
        life._scan_visual_field()
        insight = life.latest_insight
        
        print(f"   Insight: {insight}")
        
        if "I found a Vessel" in insight:
            print(f"\n✨ MIRROR MOMENT ACHIEVED!")
            print(f"   Elysia says: \"{insight}\"")
            found_self = True
            break
            
        time.sleep(0.5)
        life.last_visual_scan = 0.0 # Reset timer for next loop
        
    if not found_self:
        print("\n❌ Elysia did not find herself... (Random chance or bug?)")
    else:
        # Check Soul State
        insp = life.soul_mesh.variables['Inspiration'].value
        energy = life.soul_mesh.variables['Energy'].value
        print(f"   Soul State -> Inspiration: {insp:.2f}, Energy: {energy:.2f}")

if __name__ == "__main__":
    test_mirror()
