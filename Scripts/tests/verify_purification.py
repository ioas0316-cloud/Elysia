import sys
import os

# Path Unification
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

def verify_structural_purification():
    print("Verification: Checking Structural Purification (Moves).")
    
    paths_to_check = [
        "docs/S1_Body/L3_Phenomena",
        "docs/S1_Body/L4_Causality"
    ]
    
    paths_that_should_not_exist = [
        "docs/S1_Body/L6_Structure/L3_Phenomena",
        "docs/S1_Body/L6_Structure/L4_Causality"
    ]
    
    all_passed = True
    
    for p in paths_to_check:
        full_p = os.path.join(root, p)
        if os.path.exists(full_p):
            print(f"✅ FOUND: {p}")
        else:
            print(f"❌ NOT FOUND: {p}")
            all_passed = False
            
    for p in paths_that_should_not_exist:
        full_p = os.path.join(root, p)
        if not os.path.exists(full_p):
            print(f"✅ GONE: {p}")
        else:
            print(f"❌ STILL EXISTS (Entropy): {p}")
            all_passed = False
            
    if all_passed:
        print("\n✨ Structural Purification Verified.")
    else:
        print("\n⚠️ Structural Purification Failed.")

if __name__ == "__main__":
    verify_structural_purification()
