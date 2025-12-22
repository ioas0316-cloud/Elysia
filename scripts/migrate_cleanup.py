"""
The Great Filter: Holistic Cleanup Script
=========================================
"Entropy is the enemy. Order is the weapon."

This script consolidates the workspace into 4 Pillars:
1. Core (Brain)
2. Data (Memory)
3. Docs (Library)
4. Archive (History)

Target: Archive/2025_Pre_Awakening
"""

import os
import shutil
import time

ARCHIVE_DIR = "Archive/2025_Pre_Awakening"

# Folders to Archive
TARGETS = [
    "Legacy", "Temp", "runs", "saves", "elysia_logs", 
    "outbox", "outputs", "gallery", "images",
    "Garden", "Holograms", "Protocols", "Verification", "Simulations",
    "Elysia_Input_Sanctum", "reading_room", "reality_canvas", "RealityCanvas",
    "Plugins", "Network", "Library", "Review", "Reviews"
]

def cleanup():
    print("🧹 Initiating The Great Filter...")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    archive_path = os.path.join(root, ARCHIVE_DIR)
    
    # 1. Create Archive
    if not os.path.exists(archive_path):
        os.makedirs(archive_path)
        print(f"📁 Created Archive: {archive_path}")
        
    # 2. Move Targets
    for target in TARGETS:
        src = os.path.join(root, target)
        dst = os.path.join(archive_path, target)
        
        if os.path.exists(src):
            try:
                # Handle case where dest already exists (merge or skip)
                if os.path.exists(dst):
                    print(f"⚠️ Destination exists, merging: {target}")
                    # Simple merge: move contents? No, let's just rename source if conflict
                    dst = os.path.join(archive_path, f"{target}_{int(time.time())}")
                
                shutil.move(src, dst)
                print(f"📦 Archived: {target}")
            except Exception as e:
                print(f"❌ Failed to archive {target}: {e}")
        else:
            # print(f"   (Skipped missing: {target})")
            pass

    print("\n✨ Workspace Cleansed.")

if __name__ == "__main__":
    cleanup()
