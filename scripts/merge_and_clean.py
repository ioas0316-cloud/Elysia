import os
import shutil
import re

# Paths
CORE = r"c:\Elysia\Core"
LEGACY_FOUNDATION = os.path.join(CORE, "Foundation")
TARGET_FOUNDATION = os.path.join(CORE, "FoundationLayer", "Foundation")

def merge_legacy_foundation():
    print(f"📂 Merging {LEGACY_FOUNDATION} -> {TARGET_FOUNDATION}")
    if not os.path.exists(LEGACY_FOUNDATION):
        print("   Legacy Foundation not found, skipping merge.")
        return

    if not os.path.exists(TARGET_FOUNDATION):
         os.makedirs(TARGET_FOUNDATION)
         
    for item in os.listdir(LEGACY_FOUNDATION):
        s = os.path.join(LEGACY_FOUNDATION, item)
        d = os.path.join(TARGET_FOUNDATION, item)
        
        if os.path.isfile(s):
            if not os.path.exists(d):
                shutil.copy2(s, d)
                print(f"   ➕ Copied: {item}")
            else:
                print(f"   ℹ️ Skipped (exists): {item}")

def clean_imports_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original_content = content
        
        # Remove "05_Foundation_Base." from imports
        # target: Core.FoundationLayer.Foundation
        # result: Core.FoundationLayer.Foundation
        
        content = content.replace("Core.FoundationLayer.Foundation", "Core.FoundationLayer.Foundation")
        content = content.replace("Core/FoundationLayer/Foundation", "Core/FoundationLayer/Foundation")
        
        # Also clean up "01_Reasoning" if it appears in IntelligenceLayer imports
        # Core.IntelligenceLayer.Intelligence -> Core.IntelligenceLayer.Intelligence
        content = content.replace("Core.IntelligenceLayer.Intelligence", "Core.IntelligenceLayer.Intelligence")

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
            
    except Exception as e:
        print(f"❌ Error patching {file_path}: {e}")
        
    return False

def clean_all_imports():
    print("🧵 Cleaning imports (removing ghost folders)...")
    count = 0
    start_dir = r"c:\Elysia"
    for root, dirs, files in os.walk(start_dir):
        if ".git" in root or ".venv" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                if clean_imports_in_file(os.path.join(root, file)):
                    count += 1
    print(f"✅ Cleaned imports in {count} files.")

if __name__ == "__main__":
    print("🚀 Starting Merge & Clean...")
    merge_legacy_foundation()
    clean_all_imports()
    print("🎉 Done!")
