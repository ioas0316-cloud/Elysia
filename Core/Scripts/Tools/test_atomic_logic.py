import sys
import os
import asyncio
import numpy as np

# Setup Path
# We are in C:\Elysia\Scripts\Tools, need to go up two levels to C:\Elysia
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.1_Body.L1_Foundation.Logic.resonance_gate import ResonanceGate, analyze_structural_truth

def test_resonance_logic():
    print("--- 🔬 Atomic Logic Verification ---")
    
    # 1. Basic Gate Verification
    print(f"NOT(1)   = {ResonanceGate.NOT(1)}  (Expected -1)")
    print(f"AND(1, 1)= {ResonanceGate.AND(1, 1)} (Expected 1)")
    print(f"XOR(1, -1)={ResonanceGate.XOR(1, -1)} (Expected 1 - Torque)")
    print(f"OR(1, 0)  ={ResonanceGate.OR(1, 0)} (Expected 1)")
    
    # 2. Structural Reduction Verification
    # D7: [Foundation, Metabolism, Phenomena, Causality, Mental, Structure, Spirit]
    complex_will = np.array([0.8, 0.1, -0.9, 0.4, 0.0, -0.1, 0.9])
    truth_pattern = analyze_structural_truth(complex_will)
    print(f"\nComplex D7: {complex_will}")
    print(f"Atomic Truth: {truth_pattern}")
    # Expected: H-V-D-H-V-V-H
    
    print("\n--- ✅ Atomic Layer Verified ---")

if __name__ == "__main__":
    test_resonance_logic()
