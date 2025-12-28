
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core._01_Foundation._05_Governance.Foundation.reality_sculptor import RealitySculptor

def test_seed_freedom():
    print("\n🌱 [SAFETY TEST] Testing Seed Mutation Freedom")
    print("==============================================")
    
    sculptor = RealitySculptor()
    
    # 1. Target: A core file INSIDE the seeds directory
    # Imagine we cloned CNS to the seed
    seed_core_file = "seeds/nova/Core/Foundation/central_nervous_system.py"
    
    print(f"👉 Attempting to modify Seed Core file: {seed_core_file}")
    
    is_safe = sculptor.is_safe_to_modify(seed_core_file)
    
    if is_safe:
        print("   ✅ ALLOWED: RealitySculptor recognizes this as a Seed.")
        print("   🌱 Evolution Permitted in Sandbox.")
        return True
    else:
        print("   ❌ BLOCKED: RealitySculptor thinks this is the Original Core!")
        print("      Migration to Seed will fail.")
        return False

if __name__ == "__main__":
    test_seed_freedom()
