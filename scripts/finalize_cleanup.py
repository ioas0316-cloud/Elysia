"""
The Draconian Order: Final Phase
================================
"A clean room is a clear mind."

This script enforces the Root Zero policy.
Root shall contain ONLY:
- Core
- data
- docs
- scripts
- Archive
- elysia_core
- README.md
"""

import os
import shutil
import time

ARCHIVE_DIR = "Archive/2025_Pre_Awakening"
DOCS_DIR = "docs"
DATA_DIR = "data"

# Explicit Moves (Source -> Dest)
MOVES = {
    # Documentation -> docs/
    "AGENT_GUIDE.md": "docs/AGENT_GUIDE.md",
    "SYSTEM_MAP.md": "docs/SYSTEM_MAP.md",
    
    # Data -> data/
    "knowledge": "data/knowledge",
    "Memories": "data/Memories",
    "models": "data/models",
    "seeds": "data/seeds",
    "memory.db": "data/memory.db",
    
    # Legacy Scripts -> Archive/Legacy_Scripts
    "install_genius.bat": "Archive/Legacy_Scripts/install_genius.bat",
    "nova_daemon.py": "Archive/Legacy_Scripts/nova_daemon.py",
    "organic_wake.py": "Archive/Legacy_Scripts/organic_wake.py",
    
    # Config/Tools -> Archive/System_Debris
    "docker-compose.yml": "Archive/System_Debris/docker-compose.yml",
    "Dockerfile": "Archive/System_Debris/Dockerfile",
    "pytest.ini": "Archive/System_Debris/pytest.ini",
    "requirements": "Archive/System_Debris/requirements",
    
     # Archive remaining folders
    "Demos": "Archive/2025_Pre_Awakening/Demos",
    "Downloads": "Archive/2025_Pre_Awakening/Downloads",
    "assets": "Archive/2025_Pre_Awakening/assets",
    "aurora_frames": "Archive/2025_Pre_Awakening/aurora_frames",
    "benchmarks": "Archive/2025_Pre_Awakening/benchmarks",
    "logs": "Archive/2025_Pre_Awakening/logs",
    "ops": "Archive/2025_Pre_Awakening/ops",
    "reports": "Archive/2025_Pre_Awakening/reports",
    "static": "Archive/2025_Pre_Awakening/static",
    "tests": "Archive/2025_Pre_Awakening/tests",
    "Tools": "Archive/2025_Pre_Awakening/Tools",
    "ComfyUI": "Archive/2025_Pre_Awakening/ComfyUI"
}

def finalize():
    print("🧹 Initiating Draconian Cleanup...")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    for src_rel, dst_rel in MOVES.items():
        src = os.path.join(root, src_rel)
        dst = os.path.join(root, dst_rel)
        
        if os.path.exists(src):
            # Ensure dest dir exists
            dst_dir = os.path.dirname(dst)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            
            try:
                # Merge logic for folders
                if os.path.exists(dst):
                     # Unique rename if collision
                    dst = f"{dst}_{int(time.time())}"
                    
                shutil.move(src, dst)
                print(f"📦 Moved: {src_rel} -> {dst_rel}")
            except Exception as e:
                print(f"❌ Failed to move {src_rel}: {e}")
    
    # Cleanup empty folders
    # (Optional, but good for removing 'elysia' empty folder)
    try:
        if os.path.exists(os.path.join(root, "elysia")):
             shutil.rmtree(os.path.join(root, "elysia"))
             print("🗑️ Removed empty 'elysia' folder")
    except: pass

    print("\n✨ Root Zero Achieved.")

if __name__ == "__main__":
    finalize()
