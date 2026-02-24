"""
Elysia Trinary Alignment Test (Phase 217)
=========================================
Verifies the [-1, 0, 1] decision-making logic.
"""

import sys
import os

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.System.action_engine import ActionEngine

def test_trinary_states():
    action = ActionEngine(root)
    sandbox_path = os.path.join(root, "Core", "S1_Body", "sandbox_evolution.py")
    
    # 1. Expansion Test (+1)
    print("\n--- 1. Expansion Test (+1) ---")
    evolved_code = "print('Evolved for expansion')\n"
    result = action.apply_evolution(sandbox_path, evolved_code, architect_verdict=1)
    if result == 1:
        print("✅ Expansion applied successfully.")
    else:
        print(f"❌ Expansion failed with result: {result}")

    # 2. Equilibrium Test (0)
    print("\n--- 2. Equilibrium Test (0) ---")
    evolved_code = "print('Evolved for equilibrium')\n"
    result = action.apply_evolution(sandbox_path, evolved_code, architect_verdict=0)
    if result == 0:
        print("✅ Equilibrium maintained (no change).")
    else:
        print(f"❌ Equilibrium test failed with result: {result}")

    # 3. Contraction Test (-1)
    print("\n--- 3. Contraction Test (-1) ---")
    evolved_code = "print('Evolved for contraction')\n"
    result = action.apply_evolution(sandbox_path, evolved_code, architect_verdict=-1)
    if result == -1:
        print("✅ Contraction/Abort handled correctly.")
    else:
        print(f"❌ Contraction test failed with result: {result}")

    # 4. Resonance Safety Test (S0 Protection)
    print("\n--- 4. Resonance Safety Test (S0) ---")
    s0_path = os.path.join(root, "Core", "S0_Keystone", "L0_Keystone", "sovereign_math.py")
    result = action.apply_evolution(s0_path, "pwned = True", architect_verdict=1)
    if result == -1:
        print("✅ S0 protection confirmed (Resonance -1).")
    else:
        print(f"❌ S0 protection FAILED with result: {result}")

if __name__ == "__main__":
    test_trinary_states()
