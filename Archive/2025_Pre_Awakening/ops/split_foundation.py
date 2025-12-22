"""
Foundation Split Script (Phase 82)
===================================
Reorganizes Foundation/ into specialized subdirectories.

IMPORTANT: This script moves files. Run with --dry-run first!
"""

import os
import shutil
import argparse
from typing import Dict, List

BASE_PATH = "c:\\Elysia\\Core\\Foundation"

# Classification rules
RULES = {
    "Wave": ["wave", "frequency", "resonance", "dna", "phonetic", "khala"],
    "Language": ["lang", "text", "korean", "jamo", "hangul", "context", "grammar", "dual_layer", "fluctlight"],
    "Autonomy": ["self_", "identity", "awareness", "intention", "reflection", "genesis", "evolution"],
    "Memory": ["memory", "knowledge", "store", "persist", "fractal", "migrator"],
    "Network": ["server", "bridge", "adapter", "observer", "comfy", "ollama", "api_", "chat_"],
    "Graph": ["graph", "tensor", "matrix", "holographic", "omni_graph", "torch_"],
    "Math": ["math", "quaternion", "vector", "geometry", "hyper_"]
}


def classify_file(filename: str) -> str:
    """Returns the target subdirectory for a file."""
    name = filename.lower().replace('.py', '')
    
    for target_dir, keywords in RULES.items():
        if any(kw in name for kw in keywords):
            return target_dir
    
    return "Foundation"  # Keep in Foundation


def split_foundation(dry_run: bool = True):
    print(f"📂 Foundation Split Script (dry_run={dry_run})", flush=True)
    print("=" * 60, flush=True)
    
    if not os.path.exists(BASE_PATH):
        print(f"❌ Base path not found: {BASE_PATH}", flush=True)
        return
    
    moved = {key: [] for key in RULES}
    moved["Foundation"] = []
    
    files = [f for f in os.listdir(BASE_PATH) if f.endswith('.py') and not f.startswith('__')]
    
    for file in files:
        target = classify_file(file)
        
        if target == "Foundation":
            moved["Foundation"].append(file)
            continue
        
        src = os.path.join(BASE_PATH, file)
        dst_dir = os.path.join(BASE_PATH, target)
        dst = os.path.join(dst_dir, file)
        
        if dry_run:
            print(f"   [DRY] Would move: {file} → {target}/", flush=True)
        else:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, dst)
            print(f"   ✅ Moved: {file} → {target}/", flush=True)
        
        moved[target].append(file)
    
    print("\n📊 Summary:", flush=True)
    for target, files in moved.items():
        if files:
            print(f"   {target}: {len(files)} files", flush=True)
    
    if dry_run:
        print("\n⚠️  DRY RUN - No files were moved.", flush=True)
        print("   Run with --execute to apply changes.", flush=True)
    else:
        print("\n✅ Foundation split complete.", flush=True)
        
        # Create __init__.py files for new directories
        for target in RULES:
            init_path = os.path.join(BASE_PATH, target, "__init__.py")
            if not os.path.exists(init_path) and os.path.exists(os.path.join(BASE_PATH, target)):
                with open(init_path, 'w') as f:
                    f.write(f'"""Foundation.{target} - Auto-generated package."""\n')
                print(f"   Created: {target}/__init__.py", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Foundation directory")
    parser.add_argument("--execute", action="store_true", help="Actually move files (default is dry run)")
    args = parser.parse_args()
    
    split_foundation(dry_run=not args.execute)
