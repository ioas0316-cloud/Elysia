from enum import Enum

class InterferenceType(Enum):
    CONSTRUCTIVE = 1
    DESTRUCTIVE = 2

class SovereigntyWave:
    def __init__(self):
        self.amplitude = 1.0
        self.frequency = 60.0
    def pulse(self, dt: float):
        pass

class SovereignDecision:
    def __init__(self, passed: bool = True):
        self.is_passed = passed

class VoidState:
    def __init__(self):
        self.depth = 0.0
