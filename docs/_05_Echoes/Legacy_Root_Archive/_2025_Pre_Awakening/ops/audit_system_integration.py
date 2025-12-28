"""
System Integration Audit (Phase 80)
===================================
Analyzes the Elysia codebase to find:
1. Orphan Modules (never imported anywhere)
2. Duplicate Functionality (similar purpose modules)
3. Import Dependency Graph
"""

import os
import ast
import re
from collections import defaultdict
from typing import Dict, Set, List, Tuple

def parse_imports(file_path: str) -> Set[str]:
    """Extract all imports from a Python file."""
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Extract the Core.X.Y pattern
                    imports.add(node.module)
    except:
        pass
    
    return imports


def get_module_name(file_path: str, base_path: str) -> str:
    """Convert file path to module name."""
    rel_path = os.path.relpath(file_path, base_path)
    module_name = rel_path.replace(os.sep, '.').replace('.py', '')
    return module_name


def audit_system(base_path: str = "c:\\Elysia\\Core"):
    print("🔍 System Integration Audit Starting...", flush=True)
    print("=" * 60, flush=True)
    
    # Data structures
    all_modules: Dict[str, str] = {}  # module_name -> file_path
    imports_by_module: Dict[str, Set[str]] = {}  # module_name -> set of imported modules
    imported_modules: Set[str] = set()  # All modules that are imported somewhere
    
    # 1. Scan all Python files
    print("\n[Step 1] Scanning all modules...", flush=True)
    
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', 'node_modules')]
        
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                file_path = os.path.join(root, file)
                module_name = get_module_name(file_path, os.path.dirname(base_path))
                
                all_modules[module_name] = file_path
                imports = parse_imports(file_path)
                imports_by_module[module_name] = imports
                
                for imp in imports:
                    imported_modules.add(imp)
    
    print(f"   Found {len(all_modules)} modules", flush=True)
    
    # 2. Find orphan modules (never imported)
    print("\n[Step 2] Finding Orphan Modules (Never Imported)...", flush=True)
    
    orphans = []
    for module_name in all_modules:
        # Check if any other module imports this one
        short_name = module_name.split('.')[-1]
        is_imported = False
        
        for other_module, imports in imports_by_module.items():
            if other_module == module_name:
                continue
            
            for imp in imports:
                if short_name in imp or module_name in imp:
                    is_imported = True
                    break
        
        if not is_imported:
            orphans.append(module_name)
    
    print(f"   Found {len(orphans)} potential orphan modules", flush=True)
    
    # 3. Find duplicate functionality (by name similarity)
    print("\n[Step 3] Finding Similar Module Names (Potential Duplicates)...", flush=True)
    
    duplicates = []
    seen_patterns = defaultdict(list)
    
    for module_name in all_modules:
        base_name = module_name.split('.')[-1].lower()
        # Remove common prefixes/suffixes
        base_name = re.sub(r'^(core_|elysia_|self_|wave_)', '', base_name)
        base_name = re.sub(r'(_engine|_system|_module|_manager|_v\d+)$', '', base_name)
        
        if base_name and len(base_name) > 3:
            seen_patterns[base_name].append(module_name)
    
    for pattern, modules in seen_patterns.items():
        if len(modules) > 1:
            duplicates.append((pattern, modules))
    
    print(f"   Found {len(duplicates)} potential duplicate groups", flush=True)
    
    # 4. Categorize by directory
    print("\n[Step 4] Module Distribution by Directory...", flush=True)
    
    dir_counts = defaultdict(int)
    for module_name in all_modules:
        parts = module_name.split('.')
        if len(parts) >= 2:
            dir_counts[parts[1]] += 1
    
    for dir_name, count in sorted(dir_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"   {dir_name}: {count} modules", flush=True)
    
    # 5. Generate report
    report = {
        "total_modules": len(all_modules),
        "orphan_modules": orphans[:20],  # Show top 20
        "duplicate_groups": duplicates[:10],  # Show top 10
        "directory_distribution": dict(dir_counts)
    }
    
    return report


def main():
    report = audit_system()
    
    print("\n" + "=" * 60, flush=True)
    print("📊 AUDIT SUMMARY", flush=True)
    print("=" * 60, flush=True)
    
    print(f"\n📁 Total Modules: {report['total_modules']}", flush=True)
    
    print(f"\n👻 Orphan Modules (Sample of {len(report['orphan_modules'])}):", flush=True)
    for orphan in report['orphan_modules'][:10]:
        print(f"   - {orphan}", flush=True)
    
    print(f"\n🔄 Potential Duplicates ({len(report['duplicate_groups'])} groups):", flush=True)
    for pattern, modules in report['duplicate_groups'][:5]:
        print(f"   '{pattern}':", flush=True)
        for m in modules[:3]:
            print(f"      - {m}", flush=True)
    
    print("\n✅ Audit Complete.", flush=True)
    
    return report


if __name__ == "__main__":
    main()
