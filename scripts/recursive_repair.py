import os
import shutil
import re

ROOT_DIR = r"c:\Elysia\Core"

def recursive_rename_dirs(start_path):
    # Walk bottom-up so we don't invalidate paths as we rename parents
    for root, dirs, files in os.walk(start_path, topdown=False):
        for dir_name in dirs:
            # Pattern: 01_Name -> Name
            match = re.match(r"^(\d{2,})_([a-zA-Z_]+)$", dir_name)
            if match:
                clean_name = match.group(2)
                # Avoid collision if target exists
                old_path = os.path.join(root, dir_name)
                new_path = os.path.join(root, clean_name)
                
                print(f"📂 Renaming {dir_name} -> {clean_name}")
                
                if os.path.exists(new_path):
                    print(f"   ⚠️ Target {clean_name} exists. Merging...")
                    # Merge content
                    for item in os.listdir(old_path):
                        s = os.path.join(old_path, item)
                        d = os.path.join(new_path, item)
                        if not os.path.exists(d):
                            if os.path.isdir(s):
                                shutil.copytree(s, d)
                            else:
                                shutil.copy2(s, d)
                    shutil.rmtree(old_path)
                else:
                    os.rename(old_path, new_path)

def clean_imports_globally(start_path):
    print("🧵 global import patch...")
    # Regex to find ".Name" or ".Name" or "/Name"
    # and replace with ".Name" or "/Name"
    
    # We want to match `\.\d{2,}_([a-zA-Z_]+)` and replace with `.\1`
    pattern_dot = re.compile(r"\.\d{2,}_([a-zA-Z_]+)")
    pattern_slash = re.compile(r"/\d{2,}_([a-zA-Z_]+)")
    
    count = 0
    for root, dirs, files in os.walk(start_path):
        if ".git" in root or ".venv" in root:
            continue
            
        for file in files:
            if file.endswith(".py") or file.endswith(".bat") or file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Apply regex replacement repeatedly until stable?
                    # Python regex sub replaces all non-overlapping occurrences.
                    
                    content = pattern_dot.sub(r".\1", content)
                    content = pattern_slash.sub(r"/\1", content)
                    
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        count += 1
                except Exception as e:
                    print(f"❌ Error patching {file_path}: {e}")
    print(f"✅ Patched {count} files.")

if __name__ == "__main__":
    print("🚀 Starting Recursive Repair...")
    recursive_rename_dirs(ROOT_DIR)
    # Also scan c:\Elysia for imports
    clean_imports_globally(r"c:\Elysia")
    print("🎉 Recursive Repair Done!")
