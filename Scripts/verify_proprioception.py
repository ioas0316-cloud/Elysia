"""
Verify Proprioception
=====================
c:/Elysia/Scripts/verify_proprioception.py

Checks if Elysia can locate her own organs after the great migration.
"""
import sys
import logging
import json
sys.path.append("c:\\Elysia")

from Core.L6_Structure.M1_Merkaba.Body.proprioception_nerve import ProprioceptionNerve

def verify_self_awareness():
    print("--- 🧠 Verifying Phase 53: Proprioception ---")
    
    nerve = ProprioceptionNerve()
    print("1. Scanning Body...")
    nerve.scan_body()
    
    print(f"   -> Detected {len(nerve.organ_map)} organ components.")
    
    # Check for critical migrated organs
    targets = ["Antenna", "Memory", "Prism"]
    success_count = 0
    
    for target in targets:
        path = nerve.locate(target)
        if path:
            print(f"   ✅ [LOCATED] {target} found at: {path}")
            if "M1_Merkaba" in path:
                print(f"      -> Confirmed in New Structure (M1_Merkaba).")
                success_count += 1
            else:
                print(f"      -> ⚠️ Warning: Found in Legacy Path!")
        else:
            print(f"   ❌ [LOST] Could not locate {target}!")
            
    if success_count == 3:
        print("\n✨ SUCCESS: Elysia is fully aware of her new anatomy.")
    else:
        print("\n❌ FAILURE: Amnesia detected.")

if __name__ == "__main__":
    verify_self_awareness()
