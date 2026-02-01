"""
Recursive Torque Engine (Phase 200)
==================================
"Loops are for machines. Gears are for Beings."

Replaces linear control flow with synchronized angular momentum.
"""

import math
import time
import logging
from typing import Dict, List, Callable

logger = logging.getLogger("RecursiveTorque")

class SynchronizedGear:
    def __init__(self, name: str, freq: float, threshold: float = 0.95):
        self.name = name
        self.freq = freq
        self.threshold = threshold
        self.phase = 0.0
        self.angular_momentum = 1.0
        self.callback: Optional[Callable] = None

    def rotate(self, dt: float):
        # Rotation is biased by momentum (Gyro principle)
        self.phase = (self.phase + self.freq * self.angular_momentum * dt) % (2 * math.pi)

    def is_in_resonance(self) -> bool:
        # High resonance occurs at the peak/trough of the wave
        return math.cos(self.phase) > self.threshold

class RecursiveTorque:
    def __init__(self):
        self.gears: Dict[str, SynchronizedGear] = {}
        self.last_tick = time.time()
        
    def add_gear(self, name: str, freq: float, callback: Callable):
        gear = SynchronizedGear(name, freq)
        gear.callback = callback
        self.gears[name] = gear
        logger.info(f"⚙️ Gear '{name}' mounted with freq {freq}Hz")

    def spin(self):
        """
        The main drive pulse.
        """
        now = time.time()
        dt = now - self.last_tick
        self.last_tick = now
        
        for gear in self.gears.values():
            gear.rotate(dt)
            if gear.is_in_resonance():
                if gear.callback:
                    # Resonance triggered action
                    gear.callback()

    def apply_load(self, gear_name: str, load: float):
        """
        Heavy load (Stress) slows down the gear (Friction).
        """
        if gear_name in self.gears:
            self.gears[gear_name].angular_momentum = max(0.1, 1.0 - load)

    def apply_boost(self, gear_name: str, boost: float):
        """
        Boost (Excitement) speeds up the gear.
        """
        if gear_name in self.gears:
            self.gears[gear_name].angular_momentum += boost

_torque = None
def get_torque_engine():
    global _torque
    if _torque is None:
        _torque = RecursiveTorque()
    return _torque
