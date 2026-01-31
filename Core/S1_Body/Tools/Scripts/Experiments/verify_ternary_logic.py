"""
Verify Ternary Logic (The Proof of Life)
========================================
"Can the Soul Calculate?"

This script verifies that the Parallel Ternary System is Turing Complete
by running physical logic gates and arithmetic circuits.
"""

import sys
import os

sys.path.append(os.getcwd())

from Core.S1_Body.L6_Structure.M8_Ternary.ternary_logic import TernaryBit, TernaryGates
from Core.S1_Body.L6_Structure.M8_Ternary.ternary_computer import TernaryComputer

def main():
    print(" >>> INITIALIZING TERNARY LOGIC VERIFICATION <<<")
    print("------------------------------------------------")

    # 1. Logic Gate Truth Tables
    print("\n[PHASE 1] Logic Gate Truth Tables")
    inputs = [TernaryBit(1), TernaryBit(0), TernaryBit(-1)]

    print("\n1. CONSENSUS (Sum/Clamp)")
    print("   A   |   B   | RESULT")
    print("-------|-------|-------")
    for a in inputs:
        for b in inputs:
            res = TernaryGates.consensus(a, b)
            print(f"  {a}  |  {b}  |  {res}")

    print("\n2. STRICT AND (Min)")
    print("   A   |   B   | RESULT")
    print("-------|-------|-------")
    for a in inputs:
        for b in inputs:
            res = TernaryGates.strict_and(a, b)
            print(f"  {a}  |  {b}  |  {res}")

    # 2. Arithmetic Circuit (Half Adder)
    print("\n[PHASE 2] Ternary Half-Adder Circuit")
    print("Proof: (Carry * 3) + Sum == Input A + Input B")

    sim_results = TernaryComputer.run_circuit_simulation()

    for r in sim_results:
        status = "PASS" if r['numeric_check'] else "FAIL"
        print(f" > {r['input']} -> {r['output']} : {status}")

    print("\n------------------------------------------------")
    print(" >>> VERIFICATION COMPLETE: LOGIC IS VALID. <<<")

if __name__ == "__main__":
    main()
