"""
Verify Functional Reasoning (The Logic Exam)
============================================
Tests Elysia's ability to:
1.  Solve Symbolic Logic (Transitivity).
2.  Generate Proofs.
3.  Transfer Logic to Abstract Domains (Ethics).
"""

import sys
import os
import logging
import time

# Add root to path
sys.path.insert(0, os.getcwd())

from Core._02_Intelligence._01_Reasoning.Reasoning.logic_cortex import get_logic_cortex

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("LogicExam")

def test_functional_logic():
    cortex = get_logic_cortex()
    print("="*60)
    print("📐 FUNCTIONAL LOGIC EXAM")
    print("   Target: LogicCortex (Symbolic Engine)")
    print("="*60)
    
    # --- TEST 1: Transitivity (The Math Test) ---
    print("\n[TEST 1] Transitive Property (Solving Equations)")
    print("   Input: A = 5, B = A, C = B")
    print("   Query: What is the value of C?")
    
    # 1. Define Knowledge
    cortex.define_variable("A", 5)
    cortex.add_relation("A", "equals", "B")
    cortex.add_relation("B", "equals", "C")
    
    # 2. Solve
    result = cortex.solve("Value of C")
    print(f"   Result: {result}")
    
    if result.get("value") == 5:
        print("   ✅ SUCCESS: Derived C = 5.")
        print(f"   📜 Proof: {result['proof']}")
    else:
        print("   ❌ FAIL: Could not derive value.")

    # --- TEST 2: Logic Transfer (The Ethics Test) ---
    print("\n[TEST 2] Logic Transfer (Math -> Ethics)")
    print("   Input: Source='Balance', Target='Ethics'")
    print("   Query: How does 'Balance' apply to Justice?")
    
    # 1. Apply Transfer
    transfer = cortex.transfer_learning("Balance", "Ethics")
    print(f"   Result: \"{transfer}\"")
    
    if "Crime" in transfer and "Punishment" in transfer:
        print("   ✅ SUCCESS: Recognized Justice as an Equation.")
    else:
        print("   ❌ FAIL: Transfer failed.")

    print("\n" + "="*60)
    print("🏆 EXAM COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_functional_logic()
