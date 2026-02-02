"""
Verify Hyper Genesis (The N-Dimensional Test)
=============================================
Scripts.Experiments.verify_hyper_genesis

Verifies:
1. Creation of HyperMonads (4D default).
2. Tensor Fusion (Birth of Child).
3. Lineage Tracking (Causal Residue).
4. Spectrum Analysis (Meaning extraction).
5. Dimensional Mitosis (Expansion to 5D).
"""

import sys
import os

# Ensure Core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.S1_Body.L6_Structure.M1_Merkaba.monad_ensemble import MonadEnsemble
from Core.S1_Body.L5_Mental.spectrum_causal_engine import SpectrumCausalEngine
from Core.S1_Body.L6_Structure.M1_Merkaba.hyper_monad import AXIS_ENERGY

def run_experiment():
    print("🚀 [INIT] Initializing Hyper Merkaba Engine...")
    engine = MonadEnsemble()
    spectrum = SpectrumCausalEngine()

    # 1. Inject Parents (Thesis and Antithesis)
    # Parent A: High Mass, Static (Conservative)
    p1 = engine.inject_seed([1.0, 0.1, 0.8, 0.0])

    # Parent B: Low Mass, High Energy (Radical)
    p2 = engine.inject_seed([0.2, 0.9, -0.8, 0.0])

    print(f"\n🏛️ [PARENTS CREATED]")
    print(f"  Parent A: {spectrum.interpret(p1)}")
    print(f"  Parent B: {spectrum.interpret(p2)}")

    # 2. Process Cycle to trigger Fusion
    print("\n⚡ [PROCESSING CYCLE] Colliding Vectors...")
    stats = engine.process_cycle()
    print(f"  Stats: {stats}")

    # 3. Analyze Child
    if stats['new_births'] > 0:
        child = engine.monads[-1] # Last one is the child
        print(f"\n👶 [CHILD BORN] Monad ID {child.id}")
        print(f"  Analysis: {spectrum.interpret(child)}")
        print(f"  Lineage:  {spectrum.describe_lineage(child)}")

        # Verify Lineage Depth
        if child.tensor[3] > 0.0:
            print("  ✅ [PASS] Child has advanced Time/Depth.")
        else:
            print("  ❌ [FAIL] Child Time Axis did not advance.")
    else:
        print("\n❌ [FAIL] No Child Born. Adjust Resonance Thresholds.")
        return

    # 4. Force Dimensional Mitosis (The Singularity)
    print("\n🌌 [TRIGGERING SINGULARITY] Injecting Massive Energy...")

    # Artificially pump energy into the child to break the 4D limit
    child.tensor[AXIS_ENERGY] = 1.5 # Way above 0.95 threshold
    child.evolve(friction=1.0) # Trigger check

    print(f"  Child Dimensions after Stress: {child.dimensions}D")

    if child.dimensions > 4:
        print("  ✅ [PASS] Dimensional Mitosis Successful! 5th Dimension created.")
        print(f"  New Interpretation: {spectrum.interpret(child)}")
    else:
        print("  ❌ [FAIL] Mitosis failed. Still 4D.")

if __name__ == "__main__":
    run_experiment()
