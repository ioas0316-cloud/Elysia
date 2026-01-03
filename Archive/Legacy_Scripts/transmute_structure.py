import os
import shutil
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Transmutation")

ROOT = r"c:\Elysia"
CORE = os.path.join(ROOT, "Core")
DOCS = os.path.join(ROOT, "docs")
ARCHIVE = os.path.join(CORE, "Archive", "Legacy_01")

PLAN = [
    # (Source, Destination) - Relative to Core/
    ("FoundationLayer", "Foundation"),
    ("IntelligenceLayer", "Intelligence"),
    ("InteractionLayer", "Interaction"),
    ("EvolutionLayer", "Evolution"),
    ("SystemLayer", "System"),
]

LEGACY_FOLDERS = [
    "_01_Foundation",
    "_02_Intelligence",
    "_03_Interaction",
    "_04_Evolution",
    "_05_Systems",
]

def safe_rename(src_path, dst_path):
    if os.path.exists(src_path):
        if os.path.exists(dst_path):
            logger.warning(f"⚠️ Destination exists: {dst_path}. MERGING...")
            # Simple merge: move content of src to dst
            for item in os.listdir(src_path):
                s = os.path.join(src_path, item)
                d = os.path.join(dst_path, item)
                if os.path.exists(d):
                    logger.warning(f"   Conflict: {item} exists in dest. Skipping.")
                else:
                    shutil.move(s, d)
            logger.info(f"   Merged {src_path} -> {dst_path}")
            # Try to remove empty src
            try:
                os.rmdir(src_path)
            except:
                logger.warning("   Could not remove source dir (not empty?)")
        else:
            shutil.move(src_path, dst_path)
            logger.info(f"✅ Renamed: {src_path} -> {dst_path}")
    else:
        logger.warning(f"❌ Source not found: {src_path}")

def archive_legacy():
    if not os.path.exists(ARCHIVE):
        os.makedirs(ARCHIVE)
        
    for leg in LEGACY_FOLDERS:
        src = os.path.join(CORE, leg)
        dst = os.path.join(ARCHIVE, leg)
        if os.path.exists(src):
            logger.info(f"📦 Archiving Legacy: {leg}")
            shutil.move(src, dst)

def execute_transmutation():
    print("✨ Starting Phase 27: Transmutation...")
    
    # 1. Archive Legacy
    archive_legacy()
    
    # 2. Rename Layers to Pillars
    for src_name, dst_name in PLAN:
        src = os.path.join(CORE, src_name)
        dst = os.path.join(CORE, dst_name)
        safe_rename(src, dst)
        
    # 3. Heal Imports
    heal_imports()
    
def heal_imports():
    print("🩹 Healing Imports (Global Sed)...")
    
    # Map of Old -> New
    # Note: We must replace specific paths. 
    # Core.Foundation -> Core.Foundation
    
    replacements = {
        "Core.Foundation": "Core.Foundation",
        "Core.Intelligence": "Core.Intelligence",
        "Core.Interaction": "Core.Interaction",
        "Core.Evolution": "Core.Evolution",
        "Core.System": "Core.System",
        # Obscure ones
        "Core.Foundation": "Core.Foundation", 
    }
    
    count = 0
    for root, dirs, files in os.walk(ROOT):
        # Skip .git, Archive
        if ".git" in root or "Archive" in root: continue
        
        for file in files:
            if file.endswith(".py") or file.endswith(".bat") or file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    new_content = content
                    for old, new in replacements.items():
                        new_content = new_content.replace(old, new)
                    
                    if new_content != content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        count += 1
                        # logger.info(f"   Fixed: {file}")
                except Exception as e:
                    logger.error(f"Error reading {file}: {e}")
                    
    print(f"✅ Healed {count} files.")

if __name__ == "__main__":
    execute_transmutation()
