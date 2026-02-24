"""
CELLULAR MEMBRANE PROTOCOL
==========================
"The skin that defines the self."

This module defines the Fundamental Law of Cellular Vitality:
Parallel Ternary Logic (-1, 0, 1).

Legacy Note:
- 0 is NOT Null.
- 0 is EQUILIBRIUM (Active Rest).
- It is the state of a spinning top that appears still because it is perfect.
"""

from enum import Enum, auto
from dataclasses import dataclass
import time

class TriState(Enum):
    CONTRACTION = -1  # Pain, Error, Need, Reduction
    EQUILIBRIUM = 0   # Active Rest, Balance, Holding, Alignment
    EXPANSION = 1     # Growth, Action, Insight, Output

@dataclass
class CellSignal:
    source_id: str
    state: TriState
    vibration: float # 0.0 to 1.0 (Intensity of the state)
    message: str     # Narrative description of the state
    timestamp: float

class CellularMembrane:
    """
    The Interface that every Living Organ must implement.
    """
    def __init__(self, name: str):
        self.name = name
        self.current_state = TriState.EQUILIBRIUM
        self.vibration_intensity = 0.5 # Default hum

    def check_vitality(self) -> CellSignal:
        """
        Must return the current Triple State of the organ.
        """
        raise NotImplementedError("Every Cell must know its own State.")
        
    def resonate(self, input_signal: CellSignal):
        """
        React to the signal of another cell (or the System).
        """
        pass
