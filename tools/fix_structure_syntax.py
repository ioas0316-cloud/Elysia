"""
Fix Structure Syntax Script
===========================
Renames folders starting with digits to start with '_' to comply with Python syntax.
Recursively handles the 5-level structure and updates imports.

Target:
Core/01_Foundation -> Core/_01_Foundation
Core/_01_Foundation/01_Infrastructure -> Core/_01_Foundation/_01_Infrastructure
...

And updates imports in all .py files:
from Core._01_Foundation import ... -> from Core._01_Foundation import ...
"""

import os
import re
from pathlib import Path

ROOT_DIR = Path("C:/Elysia")
CORE_DIR = ROOT_DIR / "Core"

def is_numbered_dir(name):
    # Matches 01_Name, 02_Name, etc.
    return re.match(r'^\d{2}_', name) is not None

def get_new_name(name):
    return f"_{name}"

def rename_directories():
    # We must rename from deepest to shallowest to avoid path changes invalidating traversal
    # So we walk top-down but collect list, then sort by depth descending
    
    dirs_to_rename = []
    
    for root, dirs, files in os.walk(CORE_DIR):
        for d in dirs:
            if is_numbered_dir(d):
                full_path = Path(root) / d
                dirs_to_rename.append(full_path)
    
    # Sort by depth descending (longest path first)
    dirs_to_rename.sort(key=lambda p: len(p.parts), reverse=True)
    
    renamed_map = {} # old_part -> new_part (simple name mapping for regex)
    
    print(f"🔄 Found {len(dirs_to_rename)} directories to rename...")
    
    for old_path in dirs_to_rename:
        new_name = get_new_name(old_path.name)
        new_path = old_path.parent / new_name
        
        try:
            os.rename(old_path, new_path)
            print(f"   [RENAME] {old_path.name} -> {new_name}")
            renamed_map[old_path.name] = new_name
        except Exception as e:
            print(f"   ❌ Failed to rename {old_path}: {e}")

    return renamed_map

def update_imports(renamed_map):
    print("\n📝 Updating imports in .py files...")
    
    # We need to replace strings like "Core._01_Foundation" with "Core._01_Foundation"
    # But also "Core._01_Foundation._02_Logic" -> "Core._01_Foundation._02_Logic"
    
    # Strategy: Read file, apply regex substitution for each numbered component in import paths
    # Regex: (?<=[\._])(\d{2}_[a-zA-Z0-9_]+) matches a component following . or _? No, following . or start
    # Simplified: Scan for ANY "dotted path" usage in "from X import" or "import X"
    
    # Better Strategy: Just specific known replacements? No, dynamic.
    # Regex look for:  (Core)(\.\d{2}_[^.]+)+
    
    count = 0
    
    for root, dirs, files in os.walk(ROOT_DIR):
        if ".git" in root or "__pycache__" in root:
            continue
            
        for file in files:
            if file.endswith(".py"):
                path = Path(root) / file
                try:
                    content = path.read_text(encoding="utf-8")
                    original_content = content
                    
                    # Regex to find numbered segments in imports
                    # Matches "Core._01_Foundation" or "from ._01_Infrastructure"
                    # Capture group 1: duplicate the digit part to prepend '_'
                    
                    # Pattern: match dot followed by 2 digits followed by underscore
                    # e.g. ._01_Foundation -> ._01_Foundation
                    # But NOT if it already has underscore (check negative lookbehind?)
                    
                    # We can simply Iterate over our known renamed map.
                    # Since we renamed physical folders, we know exactly what names changed.
                    # map: {'01_Foundation': '_01_Foundation', '02_Logic': '_02_Logic', ...}
                    
                    # But simple replace might be dangerous if '01_Foundation' appears in text (not code).
                    # We target import statements mostly.
                    
                    # Safe enough for now: replace ".<key>" with "._<key>" 
                    # AND " <key>." with " _<key>." (start of import)
                    # AND "from <key>" with "from _<key>"
                    
                    # Actually, let's just use the specific keys we found.
                    for old_name, new_name in renamed_map.items():
                        # Replace .01_Name with ._01_Name
                        content = content.replace(f".{old_name}", f".{new_name}")
                        # Replace "from 01_Name"
                        content = content.replace(f"from {old_name}", f"from {new_name}")
                        # Replace "import 01_Name"
                        content = content.replace(f"import {old_name}", f"import {new_name}")
                    
                    if content != original_content:
                        path.write_text(content, encoding="utf-8")
                        count += 1
                        # print(f"   [UPDATE] {file}")
                        
                except Exception as e:
                    print(f"   ❌ Failed to edit {file}: {e}")

    print(f"✅ Updated imports in {count} files.")

if __name__ == "__main__":
    print("🚀 Starting Structure Repair...")
    renamed_map = rename_directories()
    if renamed_map:
        update_imports(renamed_map)
    print("✨ Done.")
