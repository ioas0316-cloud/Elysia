import os
import collections

ROOT = r"c:\Elysia\Core"

def analyze_structure(root_dir):
    print(f"🔍 Analyzing structure of: {root_dir}")
    
    max_depth = 0
    deepest_path = ""
    
    # Breadth stats
    dir_breadth = collections.defaultdict(int)
    
    # Duplication check
    numbered_dirs = []
    semantic_dirs = []
    
    for root, dirs, files in os.walk(root_dir):
        # Calculate depth
        rel_path = os.path.relpath(root, root_dir)
        if rel_path == ".":
            depth = 0
        else:
            depth = rel_path.count(os.sep) + 1
            
        if depth > max_depth:
            max_depth = depth
            deepest_path = rel_path
            
        # Check breadth
        dir_breadth[rel_path] = len(dirs) + len(files)
        
        # Check specific patterns
        for d in dirs:
            if "_" in d and any(c.isdigit() for c in d):
                numbered_dirs.append(os.path.join(rel_path, d))
            else:
                semantic_dirs.append(os.path.join(rel_path, d))

    print(f"\n📊 Max Depth: {max_depth}")
    print(f"📍 Deepest Path: {deepest_path}")
    
    print("\n⚠️  High Breadth Folders (> 15 items):")
    for d, count in dir_breadth.items():
        if count > 15:
            print(f"  - {d}: {count} items")
            
    print(f"\n🏷️  Numbered Directories found: {len(numbered_dirs)}")
    if len(numbered_dirs) < 20:
        for d in numbered_dirs: print(f"  - {d}")
    else:
        print(f"  (First 5): {numbered_dirs[:5]}")

if __name__ == "__main__":
    analyze_structure(ROOT)
