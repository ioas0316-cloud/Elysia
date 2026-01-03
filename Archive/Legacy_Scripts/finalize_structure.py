import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("StructureFinalizer")

CORE = r"c:\Elysia\Core"
FOUNDATION = os.path.join(CORE, "Foundation")
CORE_MEMORY = os.path.join(CORE, "Memory")
TARGET_MEMORY = os.path.join(FOUNDATION, "Memory")
NESTED_FOUNDATION = os.path.join(FOUNDATION, "Foundation")

def merge_folders(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
        
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        
        if os.path.exists(d):
            if os.path.isdir(s):
                merge_folders(s, d)
            else:
                logger.warning(f"⚠️ Conflict: {item} exists in {dst}. Overwriting.")
                shutil.move(s, d)
        else:
             shutil.move(s, d)
             # logger.info(f"   Moved {item} -> {dst}")
             
    try:
        os.rmdir(src)
        logger.info(f"✅ Removed empty source: {src}")
    except:
        pass

def fix_imports():
    print("🩹 Fixing imports for moved Memory...")
    # Core.Foundation.Memory -> Core.Foundation.Memory
    # Core.FoundationLayer.Foundation -> Core.Foundation
    
    replacements = {
        "Core.Foundation.Memory": "Core.Foundation.Memory",
        "Core.Foundation": "Core.Foundation",
        "from Core.Foundation": "from Core.Foundation",
        "import Core.Foundation": "import Core.Foundation"
    }
    
    count = 0
    for root, dirs, files in os.walk(r"c:\Elysia"):
        if ".git" in root or "Archive" in root: continue
        
        for file in files:
            if file.endswith(".py") or file.endswith(".md"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    new_content = content
                    for old, new in replacements.items():
                        new_content = new_content.replace(old, new)
                        
                    if new_content != content:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        count += 1
                except Exception as e:
                    pass
    print(f"✅ Adjusted imports in {count} files.")

def finalize():
    print("🏗️ Finalizing Structure...")
    
    # 1. Merge Core/Memory -> Core/Foundation/Memory
    if os.path.exists(CORE_MEMORY):
        print(f"   Merging {CORE_MEMORY} -> {TARGET_MEMORY}")
        merge_folders(CORE_MEMORY, TARGET_MEMORY)
        
    # 2. Flatten Core/Foundation/Foundation -> Core/Foundation
    if os.path.exists(NESTED_FOUNDATION):
        print(f"   Flattening {NESTED_FOUNDATION} -> {FOUNDATION}")
        merge_folders(NESTED_FOUNDATION, FOUNDATION)
        
    # 3. Fix Imports
    fix_imports()

if __name__ == "__main__":
    finalize()
