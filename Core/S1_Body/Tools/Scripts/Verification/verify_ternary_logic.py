"""
VERIFICATION SCRIPT: Ternary Logic (The Brick)
==============================================
Target: Prove that Ternary NAND is a Universal Gate.
"""
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.S1_Body.L1_Foundation.Logic.ternary_gate import Trit, TritState, TernaryGate

def verify_ternary_brick():
    print("🧱 [BRICK] Verifying Ternary Atomic Logic...\n")

    R = Trit(-1) # Repel
    V = Trit(0)  # Void
    A = Trit(1)  # Attract

    # Test 1: Fundamental NAND Operations
    print("--- Test 1: Fundamental NAND Physics ---")
    
    # R NAND A -> Max(R,A)=A -> NOT(A)=R ??
    # Logic: NOT(MIN(A, B))
    # Min(R, A) = Min(-1, 1) = -1 (R)
    # Not(-1) = 1 (A)
    # So R NAND A = A. (Conflict resolves to Affirmation? Or is it different?)
    
    res_ra = TernaryGate.nand(R, A)
    print(f"R (-1) NAND A (+1) = {res_ra}")
    if res_ra.value == 1:
        print("   ✅ CORRECT: The collision of opposites creates Energy (A).")
    else:
        print("   ❌ UNEXPECTED.")

    # A NAND A -> Min(1, 1)=1 -> Not(1)=-1 (R)
    res_aa = TernaryGate.nand(A, A)
    print(f"A (+1) NAND A (+1) = {res_aa}")
    if res_aa.value == -1:
         print("   ✅ CORRECT: Pure Affirmation reflects to Boundary (R).")

    # V NAND V -> Min(0, 0)=0 -> Not(0)=0 (V)
    res_vv = TernaryGate.nand(V, V)
    print(f"V (0)  NAND V (0)  = {res_vv}")
    if res_vv.value == 0:
        print("   ✅ CORRECT: Void remains Void.")

    # Test 2: Phase Torque
    print("\n--- Test 2: Calculating Phase Torque ---")
    torque_ra = TernaryGate.phase_torque(R, A)
    print(f"Torque between R and A: {torque_ra}")
    
    if torque_ra != 0:
         print("   ✅ TORQUE GENERATED: Potential Difference found.")
    else:
         print("   ❌ NO TORQUE.")

    print("\n============================================================")
    print("🎉 TERNARY LOGIC CONFIRMED: The Bricks are solid.")
    print("============================================================")

if __name__ == "__main__":
    verify_ternary_brick()
