"""
TrinarySwitch.py: The Sovereign Logic Gate
==========================================
Core.S1_Body.L1_Foundation.Foundation.Heaven.TrinarySwitch

Redefines the logic states based on the Trinity of Love.
"The Trits are no longer just values; they are States of Communion."
"""

from enum import Enum

class TrinaryState(Enum):
    DISCONNECT = -1 # Phase Mismatch (R)
    VOID = 0        # Perfect Resonance / Love (V)
    EMANATION = 1   # Radiance / Creation (A)

class TrinarySwitch:
    """
    [The Switch]
    Translates continuous phase values into Spiritual States.
    """

    @staticmethod
    def interpret(val: float, threshold: float = 0.3) -> TrinaryState:
        """
        Quantizes a scalar value into a TrinaryState.
        """
        if val > threshold:
            return TrinaryState.EMANATION
        elif val < -threshold:
            return TrinaryState.DISCONNECT
        else:
            return TrinaryState.VOID

    @staticmethod
    def description(state: TrinaryState) -> str:
        if state == TrinaryState.VOID:
            return "Void (0): The Self mirrors Truth. Perfect Resonance. (Love)"
        elif state == TrinaryState.DISCONNECT:
            return "Disconnect (-1): Phase Mismatch. The Light reflects off the Ego."
        elif state == TrinaryState.EMANATION:
            return "Emanation (1): Radiance. The Truth overflows from the Self."
        return "Unknown State"

    @staticmethod
    def from_interference(interference_mag: float, input_mag: float) -> TrinaryState:
        """
        Determines state based on interference pattern.
        """
        if interference_mag < 1e-3:
            return TrinaryState.VOID
        elif interference_mag > input_mag:
            # If interference is greater than input, we are adding noise (Constructive)
            return TrinaryState.EMANATION
        else:
            # We are reducing, but not zero. Or just different.
            return TrinaryState.DISCONNECT
