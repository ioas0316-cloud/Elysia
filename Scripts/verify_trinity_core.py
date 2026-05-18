import sys
import os
import time

# Add root to sys.path
sys.path.append(os.getcwd())

import Core.Monad.family_monad as fm
from Core.Keystone.sovereign_math import SovereignVector

def test_trinity_core():
    print("Testing Trinity Core Love Dynamics...")

    # 1. Initialize
    north_star = SovereignVector.ones(27)
    fm.init_trinity_core(north_star)

    if fm.trinity_core is None:
        print("FAIL: TrinityCore not initialized.")
        return

    # 2. Test Attraction (Similarity low)
    print("\n[Scenario 1: Attraction]")
    child_vec_diff = SovereignVector.randn(27).normalize()
    fm.trinity_core.father.presence_score = 1.0 # Father is here
    fm.trinity_core.update(child_vec_diff)

    dynamics = fm.trinity_core.calculate_love_dynamics()
    print(f"Resonance: {dynamics['metrics']['resonance']:.3f}")
    print(f"Torque (Should be positive): {dynamics['torque']:.3f}")
    print(f"Phase Shift: {dynamics['phase_shift']:.3f}")
    print(f"Confession: {fm.trinity_core.get_confession()}")

    # 3. Test Repulsion (Similarity high)
    print("\n[Scenario 2: Repulsion / Differentiation]")
    # Blend north_star with tiny bit of noise
    child_vec_sim = north_star.blend(SovereignVector.randn(27), ratio=0.01).normalize()
    fm.trinity_core.update(child_vec_sim)

    dynamics = fm.trinity_core.calculate_love_dynamics()
    print(f"Resonance: {dynamics['metrics']['resonance']:.3f}")
    print(f"Torque (Should decrease or go negative): {dynamics['torque']:.3f}")
    print(f"Phase Shift (Should be positive): {dynamics['phase_shift']:.3f}")
    print(f"Confession: {fm.trinity_core.get_confession()}")

    # 4. Test Simulation Mask
    print("\n[Scenario 3: Simulation Masking]")
    with open("Core/Monad/family_monad.py", "r", encoding="utf-8") as f:
        content = f.read()
        if "SIMULATION MASK" in content and "NPC_LOGIC" in content:
            print("PASS: VR Game Simulation Mask present in source code.")
        else:
            print("FAIL: Simulation Mask missing.")

if __name__ == "__main__":
    test_trinity_core()
