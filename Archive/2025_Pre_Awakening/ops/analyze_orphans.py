"""
Orphan Module Analyzer (Phase 82)
=================================
Categorizes the 378 orphan modules to prioritize cleanup.
"""

import os
import ast
from collections import defaultdict
from typing import Dict, List, Set

def categorize_orphans(base_path: str = "c:\\Elysia\\Core"):
    print("🔍 Analyzing Orphan Modules...", flush=True)
    print("=" * 60, flush=True)
    
    # 1. Build import graph
    all_modules: Dict[str, str] = {}
    imports_by_module: Dict[str, Set[str]] = {}
    
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', 'node_modules')]
        
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, os.path.dirname(base_path))
                module_name = rel_path.replace(os.sep, '.').replace('.py', '')
                
                all_modules[module_name] = file_path
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    tree = ast.parse(content)
                    
                    imports = set()
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.add(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.add(node.module)
                    
                    imports_by_module[module_name] = imports
                except:
                    imports_by_module[module_name] = set()
    
    # 2. Find orphans (never imported)
    imported_anywhere = set()
    for module, imports in imports_by_module.items():
        for imp in imports:
            imported_anywhere.add(imp)
            # Also match short names
            for mod_name in all_modules:
                if mod_name.endswith(imp) or imp in mod_name:
                    imported_anywhere.add(mod_name)
    
    orphans = [m for m in all_modules if m not in imported_anywhere]
    
    # 3. Categorize orphans by type
    categories = {
        "demos_tests": [],      # Demo/test files
        "entry_points": [],     # Main scripts (__main__, run_, start_)
        "legacy": [],           # Old/deprecated
        "engines": [],          # *_engine.py
        "utils": [],            # *_utils.py, helpers
        "unknown": []           # Unknown purpose
    }
    
    for orphan in orphans:
        short_name = orphan.split('.')[-1].lower()
        path = all_modules[orphan]
        
        # Categorize
        if any(x in short_name for x in ['demo', 'test', 'verify', 'benchmark']):
            categories["demos_tests"].append(orphan)
        elif any(x in short_name for x in ['run_', 'start_', 'main', 'launch']):
            categories["entry_points"].append(orphan)
        elif any(x in short_name for x in ['legacy', 'old', 'deprecated', 'backup']):
            categories["legacy"].append(orphan)
        elif '_engine' in short_name or short_name.endswith('engine'):
            categories["engines"].append(orphan)
        elif any(x in short_name for x in ['util', 'helper', 'common']):
            categories["utils"].append(orphan)
        else:
            categories["unknown"].append(orphan)
    
    # 4. Print results
    print(f"\n📊 Categorized {len(orphans)} Orphan Modules:\n", flush=True)
    
    for category, modules in categories.items():
        print(f"📁 {category.upper()}: {len(modules)} modules", flush=True)
        for m in modules[:5]:
            print(f"   - {m}", flush=True)
        if len(modules) > 5:
            print(f"   ... and {len(modules) - 5} more", flush=True)
        print()
    
    # 5. Recommendations
    print("💡 RECOMMENDATIONS:", flush=True)
    print(f"   1. KEEP: {len(categories['entry_points'])} entry points (intended to run directly)", flush=True)
    print(f"   2. MOVE to Demos/: {len(categories['demos_tests'])} demo/test files", flush=True)
    print(f"   3. ARCHIVE: {len(categories['legacy'])} legacy files", flush=True)
    print(f"   4. REVIEW: {len(categories['unknown'])} unknown purpose files", flush=True)
    
    return categories


def analyze_foundation(base_path: str = "c:\\Elysia\\Core\\Foundation"):
    """Analyzes Foundation directory for split recommendations."""
    print("\n" + "=" * 60, flush=True)
    print("📂 Analyzing Foundation Directory...", flush=True)
    print("=" * 60, flush=True)
    
    categories = defaultdict(list)
    
    for file in os.listdir(base_path):
        if not file.endswith('.py') or file.startswith('__'):
            continue
        
        name = file.lower().replace('.py', '')
        
        # Categorize by prefix/suffix patterns
        if any(x in name for x in ['wave', 'frequency', 'resonance']):
            categories["Wave/"].append(file)
        elif any(x in name for x in ['self_', 'identity', 'awareness']):
            categories["Autonomy/"].append(file)
        elif any(x in name for x in ['graph', 'tensor', 'matrix', 'torch']):
            categories["Graph/"].append(file)
        elif any(x in name for x in ['lang', 'text', 'korean', 'jamo', 'phonetic']):
            categories["Language/"].append(file)
        elif any(x in name for x in ['memory', 'knowledge', 'store', 'persist']):
            categories["Memory/"].append(file)
        elif any(x in name for x in ['math', 'quaternion', 'vector', 'geometry']):
            categories["Math/"].append(file)
        elif any(x in name for x in ['comfy', 'ollama', 'bridge', 'adapter', 'server']):
            categories["Network/"].append(file)
        else:
            categories["Foundation/ (keep)"].append(file)
    
    print("\n📊 Proposed Split:\n", flush=True)
    
    for target_dir, files in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"📁 {target_dir}: {len(files)} files", flush=True)
        for f in files[:5]:
            print(f"   - {f}", flush=True)
        if len(files) > 5:
            print(f"   ... and {len(files) - 5} more", flush=True)
        print()
    
    return dict(categories)


if __name__ == "__main__":
    orphan_cats = categorize_orphans()
    foundation_split = analyze_foundation()
    
    print("\n✅ Analysis Complete.", flush=True)
