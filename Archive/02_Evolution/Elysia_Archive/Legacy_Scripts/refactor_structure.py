import os
import shutil
import re

# Configuration
CORE_DIR = r"c:\Elysia\Core"
MAPPINGS = {
    "01_Foundation": "FoundationLayer",
    "02_Intelligence": "IntelligenceLayer",
    "03_Interaction": "InteractionLayer",
    "04_Evolution": "EvolutionLayer",
    "05_Systems": "SystemLayer"
}

def rename_folders():
    print("📂 Renaming folders...")
    for old_name, new_name in MAPPINGS.items():
        old_path = os.path.join(CORE_DIR, old_name)
        new_path = os.path.join(CORE_DIR, new_name)
        
        if os.path.exists(old_path):
            try:
                if os.path.exists(new_path):
                     print(f"⚠️ Target {new_name} already exists. Merging...")
                     # Simple merge: move content from old to new
                     for item in os.listdir(old_path):
                         s = os.path.join(old_path, item)
                         d = os.path.join(new_path, item)
                         if os.path.exists(d):
                             print(f"   Skipping {item} (exists in target)")
                         else:
                             shutil.move(s, d)
                     os.rmdir(old_path) # Remove if empty
                else:
                    os.rename(old_path, new_path)
                    print(f"✅ Renamed: {old_name} -> {new_name}")
            except Exception as e:
                print(f"❌ Error renaming {old_name}: {e}")
        else:
            print(f"ℹ️ {old_name} not found (maybe already renamed?)")

def patch_imports_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original_content = content
        
        # Regex replacement for "from Core.Foundation" etc.
        # Also handles "import Core.Foundation"
        for old_name, new_name in MAPPINGS.items():
            # Pattern 1: Dot notation (from Core.Foundation)
            pattern_dot = f"Core\.{old_name}"
            repl_dot = f"Core.{new_name}"
            content = re.sub(pattern_dot, repl_dot, content)
            
            # Pattern 2: Slash notation in comments or strings (Core/FoundationLayer)
            pattern_slash = f"Core/{old_name}"
            repl_slash = f"Core/{new_name}"
            content = re.sub(pattern_slash, repl_slash, content)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
            
    except Exception as e:
        print(f"❌ Error reading/writing {file_path}: {e}")
        
    return False

def patch_all_imports():
    print("🧵 Patching imports in all .py files...")
    count = 0
    # Walk through c:\Elysia
    start_dir = r"c:\Elysia"
    for root, dirs, files in os.walk(start_dir):
        # Skip .git, .venv etc
        if ".git" in root or ".venv" in root:
            continue
            
        for file in files:
            if file.endswith(".py") or file.endswith(".bat") or file.endswith(".md"):
                file_path = os.path.join(root, file)
                if patch_imports_in_file(file_path):
                    print(f"   Patched: {file}")
                    count += 1
    print(f"✅ Patched {count} files.")

if __name__ == "__main__":
    print("🚀 Starting Structural Repair...")
    try:
        rename_folders()
        patch_all_imports()
        print("🎉 Repair Complete!")
    except Exception as e:
        print(f"🔥 FATAL ERROR: {e}")
