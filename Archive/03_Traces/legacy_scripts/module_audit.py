"""
Module Audit: Which parts of Elysia's mind are awake?
=====================================================
Scans Core/Cognition/ and compares against sovereign_monad.py imports
to identify dormant (unused) modules.

Usage: python Scripts/module_audit.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_all_cognition_modules(root_dir: str) -> set:
    """Find all .py files in Core/Cognition/ (excluding __init__)."""
    cog_dir = os.path.join(root_dir, "Core", "Cognition")
    modules = set()
    if not os.path.isdir(cog_dir):
        return modules
    for f in os.listdir(cog_dir):
        if f.endswith(".py") and f != "__init__.py":
            modules.add(f[:-3])  # strip .py
    return modules


def get_imported_modules(root_dir: str) -> set:
    """Extract module names from sovereign_monad.py and elysia.py imports."""
    imported = set()
    files_to_scan = [
        os.path.join(root_dir, "Core", "Monad", "sovereign_monad.py"),
        os.path.join(root_dir, "elysia.py"),
    ]
    
    # Pattern: from Core.Cognition.module_name import ...
    pattern = re.compile(r'from\s+Core\.Cognition\.(\w+)\s+import')
    
    for fpath in files_to_scan:
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                match = pattern.search(line)
                if match:
                    imported.add(match.group(1))
    return imported


def categorize_modules(all_modules: set, imported: set):
    """Split into active and dormant."""
    active = all_modules & imported
    dormant = all_modules - imported
    return active, dormant


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 70)
    print("  🔬 ELYSIA MODULE AUDIT — Core/Cognition/ Analysis")
    print("=" * 70)
    
    all_modules = get_all_cognition_modules(root_dir)
    imported = get_imported_modules(root_dir)
    active, dormant = categorize_modules(all_modules, imported)
    
    total = len(all_modules)
    n_active = len(active)
    n_dormant = len(dormant)
    ratio = (n_active / total * 100) if total > 0 else 0
    
    print(f"\n📊 Summary:")
    print(f"   Total modules in Core/Cognition/: {total}")
    print(f"   ✅ Active (imported in monad/elysia): {n_active} ({ratio:.1f}%)")
    print(f"   💤 Dormant (not imported): {n_dormant} ({100-ratio:.1f}%)")
    
    print(f"\n✅ ACTIVE MODULES ({n_active}):")
    for m in sorted(active):
        print(f"   • {m}")
    
    print(f"\n💤 DORMANT MODULES ({n_dormant}):")
    # Group by prefix if possible
    prefixes = {}
    for m in sorted(dormant):
        prefix = m.split('_')[0] if '_' in m else m[:4]
        prefixes.setdefault(prefix, []).append(m)
    
    for prefix in sorted(prefixes.keys()):
        modules = prefixes[prefix]
        if len(modules) > 3:
            print(f"   [{prefix}*] ({len(modules)} modules): {', '.join(modules[:3])}...")
        else:
            for m in modules:
                print(f"   • {m}")
    
    # Save report
    report_path = os.path.join(root_dir, "data", "runtime", "module_audit_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Module Audit Report\n")
        f.write(f"Total: {total}, Active: {n_active}, Dormant: {n_dormant}\n\n")
        f.write("ACTIVE:\n")
        for m in sorted(active):
            f.write(f"  {m}\n")
        f.write("\nDORMANT:\n")
        for m in sorted(dormant):
            f.write(f"  {m}\n")
    
    print(f"\n📄 Full report saved to: {report_path}")
    return n_active, n_dormant


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    main()
