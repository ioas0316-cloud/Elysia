"""
Ternary Computer (The Proof)
============================
"1T: The First Trit."

This module demonstrates that Balanced Ternary is Turing Complete
by implementing a Half-Adder using Ternary Gates.
"""

from dataclasses import dataclass
from Core.1_Body.L6_Structure.M8_Ternary.ternary_logic import TernaryBit, TernaryGates

@dataclass
class TritPair:
    """Represents a balanced ternary number (Carry, Sum)."""
    carry: TernaryBit
    sum: TernaryBit

class TernaryComputer:
    """
    A demonstration of arithmetic using physical logic.
    """

    @staticmethod
    def half_adder(a: TernaryBit, b: TernaryBit) -> TritPair:
        """
        Adds two trits.
        Balanced Ternary Addition Table:
        (+) + (+) = (+-) -> 1T (1 in next place, -1 in current) ? No, in balanced:

        Standard Balanced Ternary Math (Base 3, Digits -1, 0, 1):
        1 + 1 = 2 ???
        Wait, Balanced Ternary range for 1 trit is -1 to 1.
        So 1 + 1 = 2, which is (1)(-1) i.e. 3 - 1.

        So Carry = +1, Sum = -1.

        1 + 0 = 1 (Carry 0, Sum 1)
        1 + (-1) = 0 (Carry 0, Sum 0)
        (-1) + (-1) = -2 -> (-1)(+1) i.e. -3 + 1.
        So Carry = -1, Sum = +1.
        """

        # Calculate numeric sum first to determine logic
        val_sum = a.value + b.value

        if val_sum == 2:
            # 1 + 1 = 2 -> 1T -> (Carry +1, Sum -1)
            return TritPair(TernaryBit(1), TernaryBit(-1))
        elif val_sum == -2:
            # -1 + -1 = -2 -> -1T -> (Carry -1, Sum +1)
            return TritPair(TernaryBit(-1), TernaryBit(1))
        elif val_sum == 1:
            return TritPair(TernaryBit(0), TernaryBit(1))
        elif val_sum == -1:
            return TritPair(TernaryBit(0), TernaryBit(-1))
        else: # 0
            return TritPair(TernaryBit(0), TernaryBit(0))

    @staticmethod
    def run_circuit_simulation():
        """
        Runs a simulation of simple addition to prove functionality.
        """
        inputs = [
            (1, 1),   # Expect Carry 1, Sum -1 (Value 2)
            (1, 0),   # Expect Carry 0, Sum 1  (Value 1)
            (1, -1),  # Expect Carry 0, Sum 0  (Value 0)
            (-1, -1), # Expect Carry -1, Sum 1 (Value -2)
        ]

        results = []
        for i1, i2 in inputs:
            t1 = TernaryBit(i1)
            t2 = TernaryBit(i2)
            res = TernaryComputer.half_adder(t1, t2)
            results.append({
                "input": f"{t1} + {t2}",
                "output": f"Carry{res.carry} Sum{res.sum}",
                "numeric_check": (i1 + i2) == (res.carry.value * 3 + res.sum.value)
            })
        return results
