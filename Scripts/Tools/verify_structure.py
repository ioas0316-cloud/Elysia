import os
import sys
from pathlib import Path

def verify_structure():
    root = Path("c:/Elysia")
    core = root / "Core"
    
    expected_layers = ["L0_Keystone", "L1_Foundation", "L2_Metabolism", "L3_Phenomena", 
                       "L4_Causality", "L5_Mental", "L6_Structure", "L7_Spirit"]
    
    print(f"🔍 Verifying Elysia Structure at {core}")
    
    for layer in expected_layers:
        layer_path = core / layer
        if layer_path.exists():
            print(f"  [OK] {layer} found.")
        else:
            print(f"  [!!] {layer} MISSING!")

    # Check for the 'Foundation Dump'
    foundation_dump = core / "L1_Foundation" / "Foundation"
    if foundation_dump.exists():
        files = list(foundation_dump.glob("*.py"))
        print(f"  [!] Foundation Dump contains {len(files)} files. Migration required.")
        if len(files) > 0:
            for f in files[:5]: # Show first 5
                print(f"    - {f.name}")
    
if __name__ == "__main__":
    verify_structure()
