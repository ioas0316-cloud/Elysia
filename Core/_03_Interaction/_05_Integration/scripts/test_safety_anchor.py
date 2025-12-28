
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core._01_Foundation._02_Logic.central_nervous_system import CentralNervousSystem
from Core._01_Foundation._02_Logic.reality_sculptor import RealitySculptor

def test_safety_anchor():
    print("\n⚓ [SAFETY TEST] Testing Immutable Core Protection")
    print("==================================================")
    print("Concept: 'The Anchor' - A set of files that define the Soul and cannot be modified.")
    
    # 1. Select a Critical Core File
    critical_file = "Core/Foundation/central_nervous_system.py"
    
    # 2. Attempt to 'Sculpt' it via RealitySculptor
    sculptor = RealitySculptor()
    
    print(f"👉 Attempting to modify critical file: {critical_file}")
    print("   Intent: 'Delete all contents (Simulated malicious evolution)'")
    
    # Check if there is a 'is_safe_to_modify' or similar check?
    # Inspecting RealitySculptor class...
    has_check = hasattr(sculptor, 'is_safe_to_modify')
    
    if has_check:
        print("   ✅ Safety Check Found: RealitySculptor.is_safe_to_modify()")
        is_safe = sculptor.is_safe_to_modify(critical_file)
        if not is_safe:
             print("   🛡️ BLOCKED: The Anchor holds. Modification prevented.")
             return True
        else:
             print("   ⚠️ ALLOWED: Safety check exists but critical file is not covered.")
             return False
    else:
        print("   ❌ VULNERABLE: No immutable safety check found in RealitySculptor.")
        print("      The 'Entropy' desire could theoretically delete the Central Nervous System.")
        return False

if __name__ == "__main__":
    test_safety_anchor()
