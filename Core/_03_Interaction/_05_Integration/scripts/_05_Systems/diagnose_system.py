"""
System Diagnostician: Full Body Scan
====================================

"To heal the patient, we must first find the wound."

Purpose:
Recursively scan the 'Core/' directory and attempt to import every module.
Report:
- Syntax Errors (Broken Code)
- Import Errors (Broken Paths)
- Runtime Errors (Initialization Failures)
"""

import os
import sys
import importlib.util
import traceback

# Add root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def diagnose():
    print("🏥 Starting System Diagnostic Scan...")
    print(f"   Root: {os.getcwd()}")
    
    scan_dir = os.path.abspath("Core")
    error_count = 0
    checked_count = 0
    
    for root, dirs, files in os.walk(scan_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                checked_count += 1
                file_path = os.path.join(root, file)
                module_name = _path_to_module(file_path)
                
                try:
                    # Attempt Import
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        # print(f"   ✅ {module_name}") # Too verbose
                except Exception as e:
                    error_count += 1
                    print(f"\n❌ FAILURE: {module_name}")
                    print(f"   Path: {file_path}")
                    print(f"   Error: {str(e)}")
                    # print(traceback.format_exc()) # Optional: Show trace

    print("\n" + "="*40)
    print(f"📊 Diagnostic Result")
    print(f"   Modules Scanned: {checked_count}")
    print(f"   Errors Found:    {error_count}")
    
    if error_count == 0:
        print("✅ System Health: PRISTINE (100%)")
    else:
        print(f"⚠️ System Health: COMPROMISED ({error_count} fractures)")

def _path_to_module(path):
    rel = os.path.relpath(path, os.getcwd())
    return rel.replace(os.path.sep, ".").replace(".py", "")

if __name__ == "__main__":
    diagnose()
