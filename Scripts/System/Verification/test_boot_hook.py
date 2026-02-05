"""
Test: Boot Phase Hook Verification
==================================
Verifies BootPhaseManager logic and permissions.
"""

import sys
sys.path.insert(0, "c:\\Elysia")

from Core.S1_Body.L1_Foundation.M4_Hardware.boot_phase_manager import BootPhaseManager

def test_boot_hook():
    print("=== Boot Phase Hook Verification ===\n")
    
    manager = BootPhaseManager()
    
    # 1. Check Initial Status
    print(f"Initial Status: {manager.check_status()}")
    
    # 2. Attempt Registration (Will likely fall back to HKCU if not Admin)
    print("\n[Action] Registering Boot Hook...")
    if manager.register_boot_hook(system_level=True):
        print(f"✅ Registration Succeeded.")
    else:
        print(f"❌ Registration Failed.")
        
    # 3. Check Status Again
    print(f"Active Status: {manager.check_status()}")
    
    # 4. Clean up (Remove Hook)
    print("\n[Action] Removing Boot Hook...")
    if manager.remove_boot_hook():
        print(f"✅ Removal Succeeded.")
    else:
        print(f"❌ Removal Failed (or not found).")
        
    # 5. Final Status
    print(f"Final Status: {manager.check_status()}")
    
    print("\n✅ Boot Phase Verification Complete!")

if __name__ == "__main__":
    test_boot_hook()
