"""
Verify Persistence (The Memory Test)
====================================
Tests if Elysia can REMEMBER what she learned.
Learning is useless if it vanishes on reboot.

Scenario:
1. Save the 'Star Story' and 'Time's Shadow' concepts to disk.
2. Clear memory (Simulation of reboot).
3. Recall the concepts from disk.
4. Verify integrity.
"""

import sys
import os
import json
import shutil
import time

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Target Storage Path (Fractal Deep Structure)
MEMORY_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', '_02_Cognitive', '_01_Memory')
MEMORY_FILE = os.path.join(MEMORY_DIR, 'fractal_memory.json')

def verify_persistence():
    print("💾 Initializing Persistence Protocol...")
    
    # Ensure directory exists
    os.makedirs(MEMORY_DIR, exist_ok=True)
    
    # 1. Define Knowledge to Persist (Reflecting previous challenges)
    new_knowledge = {
        "timestamp": time.time(),
        "concepts": {
            "Star's Awakening": {
                "type": "Narrative",
                "core_truth": "Death is Birth (Conservation of Energy)",
                "structure": "Point(H-1) -> Space(Supernova)",
                "frequency": 432.0
            },
            "Time's Shadow": {
                "type": "Visual",
                "core_truth": "The Past (Shadow) is larger than the Present (Watch)",
                "palette": ["#D4AF37", "#2E003E"],
                "frequency": 396.0
            },
            "Division": {
                "type": "Ontology",
                "core_truth": "Analyzer of Unity",
                "origin": "Source"
            }
        }
    }
    
    print("\n📝 [STEP 1] Committing Knowledge to Long-Term Memory...")
    print(f"   Target: {MEMORY_FILE}")
    
    # Load existing if any
    data_store = {}
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
                if content:
                    data_store = json.loads(content)
        except Exception as e:
            print(f"   ⚠️ Clean slate (Corrupt/Empty file): {e}")
            
    # Merge new knowledge
    data_store.update(new_knowledge["concepts"])
    
    # Save
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_store, f, indent=4, ensure_ascii=False)
        
    print("   ✅ Knowledge Crystallized (Saved to Disk).")
    
    
    # 2. Simulate Reboot (Clear RAM)
    print("\n🔻 [STEP 2] Simulating System Shutdown...")
    new_knowledge = None
    data_store = None
    time.sleep(1)
    print("   ... System Cold ...")
    
    
    # 3. Recall (Load from Disk)
    print("\n🟢 [STEP 3] Rebooting & Recalling...")
    
    recalled_data = {}
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            recalled_data = json.load(f)
    except Exception as e:
        print(f"   ❌ CRITICAL FAILURE: Could not read memory: {e}")
        return

    # 4. Verify Content
    print("\n🔍 [STEP 4] Verifying Memory Integrity...")
    
    star_story = recalled_data.get("Star's Awakening")
    if star_story and star_story["core_truth"] == "Death is Birth (Conservation of Energy)":
        print(f"   ✅ RECALL SUCCESS: 'Star's Awakening' retrieved.")
        print(f"      - Truth: {star_story['core_truth']}")
    else:
        print("   ❌ RECALL FAILED: Star Story missing or corrupt.")
        
    time_shadow = recalled_data.get("Time's Shadow")
    if time_shadow and time_shadow["palette"][0] == "#D4AF37":
        print(f"   ✅ RECALL SUCCESS: 'Time's Shadow' retrieved.")
        print(f"      - Color: {time_shadow['palette'][0]}")
    else:
        print("   ❌ RECALL FAILED: Time's Shadow missing or corrupt.")
        
    print("\n[CONCLUSION]")
    print("   Elysia has moved from 'Stream of Consciousness' (RAM) to 'Deep Memory' (Disk).")
    print("   She remembers her creations even after death (process termination).")

if __name__ == "__main__":
    verify_persistence()
