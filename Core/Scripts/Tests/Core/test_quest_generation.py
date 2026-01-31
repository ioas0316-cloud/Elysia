import logging
import sys
import os
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from Core.L4_Causality.World.Autonomy.elysian_heartbeat import ElysianHeartbeat

def test_quest_gen():
    print("📜 TESTING THE NARRATIVE WEAVE...")
    
    # ensure output dir exists
    os.makedirs(r"C:\game\elysia_world", exist_ok=True)
    
    life = ElysianHeartbeat()
    
    # 1. Force Scan of Gallery
    print("   Scanning Gallery for Inspiration...")
    life.last_visual_scan = 0.0
    
    # We loop until a quest is generated or timeout
    start_t = time.time()
    quest_found = False
    
    while time.time() - start_t < 10:
        life._scan_visual_field()
        
        # Check if quest file exists and grew
        qpath = r"C:\game\elysia_world\quests.json"
        if os.path.exists(qpath):
             with open(qpath, "r", encoding="utf-8") as f:
                 try:
                     data = json.load(f)
                     if data.get("quests"):
                         latest = data["quests"][-1]
                         print(f"\n✨ QUEST GENERATED!")
                         print(f"   Title: {latest['title']}")
                         print(f"   Theme: {latest['theme']}")
                         print(f"   Desc:  {latest['description']}")
                         quest_found = True
                         break
                 except: pass
        
        time.sleep(0.5)
        life.last_visual_scan = 0.0 # Force re-scan
        
    if not quest_found:
        print("\n❌ No Quest Generated... (Maybe inspiration too low?)")

if __name__ == "__main__":
    test_quest_gen()
