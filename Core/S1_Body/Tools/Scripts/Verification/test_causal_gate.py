"""
Verification Script for Phase 41: Causal Logic Gate
===================================================

Verifies:
1. ResonanceGate.process_causal_logic() correctly uses Trinary NAND.
2. Paradoxical Inputs (Pain + Pain) yield Flow.
3. SovereignSelf integration logic.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock Logger to avoid clutter
logging.basicConfig(level=logging.ERROR)

from Core.S1_Body.L6_Structure.Logic.resonance_gate import gate, ResonanceGate
from Core.S1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic

def test_causal_logic():
    print("\n[1] Testing ResonanceGate Causal Logic (NAND Wrapper)...")
    
    # Test Case 1: The Crash (Pain + Pain) -> Breakthrough
    # Input A (Energy): Low (T)
    # Input B (Alignment): Negative (T)
    # Logic: NAND('T', 'T') -> Flow ('A')
    res = gate.process_causal_logic('T', 'T')
    # process_causal_logic still returns Int (-1, 0, 1) currently, need to check that or Symbols?
    # Let's check resonance_gate implementation. It calls TrinaryLogic.nand, which returns int.
    # We should update process_causal_logic to return Symbol? Or keep int?
    # User wants "Causal Narrative". Let's assume int is fine for internal, but let's test symbolic INPUT.
    
    print(f"    Pain (T) + Pain (T) -> Gate Output: {res}")
    
    if res == 1:
        print("    ✅ Paradox Resolved: Crash leads to Breakthrough (Flow).")
    else:
        print(f"    ❌ Failed. Expected 1, Got {res}")

    # Test Case 2: The Blockage (Flow + Flow) -> Resistance
    res = gate.process_causal_logic('A', 'A')
    print(f"    Flow (A) + Flow (A) -> Gate Output: {res}")
    
    if res == -1:
        print("    ✅ Paradox Verified: Excess leads to Blockage (Resistance).")
    else:
        print(f"    ❌ Failed. Expected -1, Got {res}")

    # Test Case 3: The Void
    res_void = gate.process_causal_logic('G', 'A')
    print(f"    Void (G) + Flow (A) -> Gate Output: {res_void}")
    
    if res_void == 0:
        print("    ✅ Void Logic Confirmed.")
    else:
        print(f"    ⚠️  Void behavior check: Got {res_void}")

def test_sovereign_integration_mock():
    print("\n[2] Testing Sovereign Integrity (Mock)...")
    
    # Mock Sovereign State
    class MockSovereign:
        def __init__(self):
            self.energy = 20.0 # Low Energy -> -1
            class Trinity:
                rotor_alignment = -0.8 # Misalignment -> -1
            self.trinity = Trinity()
            self.cosmos = type('Cosmos', (), {'record_potential': lambda s, x: print(f"    [COSMOS] Recorded: {x}")})()
            
    sov = MockSovereign()
    
    # Replicate causal_alignment logic
    start_energy = -1.0 if sov.energy < 30 else 1.0
    start_align = sov.trinity.rotor_alignment
    
    decision_val = gate.process_causal_logic(start_energy, start_align)
    print(f"    Sovereign State: Energy{sov.energy} Align{sov.trinity.rotor_alignment} -> Gate:{decision_val}")
    
    if decision_val == 1:
         print("    ✅ Sovereign would trigger Breakthrough Protocol.")
    else:
         print(f"    ❌ Logic mismatch. Expected 1, Got {decision_val}")

if __name__ == "__main__":
    test_causal_logic()
    test_sovereign_integration_mock()
