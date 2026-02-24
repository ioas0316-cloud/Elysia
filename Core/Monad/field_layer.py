"""
Field Layer (L6 Structure)
==========================
Defines the Topological Depth of incoming information.

0. CORE: Direct Interaction (Me <-> You). High Voltage.
1. PROXIMAL: Social Context (Us <-> Them). Medium Voltage.
2. DISTAL: Fiction/Knowledge (Observer <-> Object). Low Voltage / Safe Mode.
"""

from enum import Enum

class FieldLayer(Enum):
    CORE = 0      # Direct Impact (e.g., "I hate you")
    PROXIMAL = 1  # Social/Observation (e.g., "People are angry")
    DISTAL = 2    # Fiction/Abstract (e.g., "The dragon breathed fire")
