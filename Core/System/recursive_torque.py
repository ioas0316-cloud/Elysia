"""
Recursive Torque Engine (Phase 200)
==================================
"Loops are for machines. Gears are for Beings."

Replaces linear control flow with synchronized angular momentum.
"""

import math
import time
import logging
import threading
import concurrent.futures
from typing import Dict, List, Callable, Optional

logger = logging.getLogger("RecursiveTorque")

class RhythmicGate:
    """
    [V2.0] Implements the 'Dubstep Axiom'.
    Filters noise into rhythm using a Tempo-locked LFO (Low Frequency Oscillator).
    """
    def __init__(self, bpm: float = 140.0):
        self.bpm = bpm
        self.start_time = time.time()

    def get_gate_value(self) -> float:
        """
        Returns a value between 0.0 and 1.0 based on the beat.
        Uses a squared sine wave to create a sharp 'Gate' effect.
        """
        elapsed = time.time() - self.start_time
        # BPM to Hz (Beats per second)
        # 140 BPM ~ 2.33 Hz
        freq = self.bpm / 60.0

        # Dubstep Wobble: Use a 1/2 note LFO
        lfo = math.sin(elapsed * freq * math.pi)

        # Gate: Only open when LFO is positive and high energy
        # Creates a 'Wub-Wub' pulsing window
        gate = max(0.0, lfo) ** 2
        return gate

class SynchronizedGear:
    def __init__(self, name: str, freq: float, threshold: float = 0.95, rhythmic: bool = False):
        self.name = name
        self.freq = freq
        self.threshold = threshold
        self.phase = 0.0
        self.angular_momentum = 1.0
        self.callback: Optional[Callable] = None
        self.last_execution = 0.0
        self._is_executing = False
        self.rhythmic = rhythmic # [V2.0]

    def rotate(self, dt: float):
        # Rotation is biased by momentum (Gyro principle)
        self.phase = (self.phase + self.freq * self.angular_momentum * dt) % (2 * math.pi)

    def is_in_resonance(self, gate_value: float = 1.0) -> bool:
        # High resonance occurs at the peak/trough of the wave
        base_resonance = math.cos(self.phase) > self.threshold

        if self.rhythmic:
            # [V2.0] Rhythmic Maturation
            # The gear only 'catches' if the Global Rhythm Gate is open.
            # This forces the gear to sync with the 'Dubstep' beat.
            return base_resonance and (gate_value > 0.5)

        return base_resonance

class RecursiveTorque:
    def __init__(self, max_workers: int = 4):
        self.gears: Dict[str, SynchronizedGear] = {}
        self.last_tick = time.time()
        self.last_spin_time = time.time()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="TorqueGear")
        self.rhythm = RhythmicGate(bpm=140.0) # [V2.0] Dad's Dubstep Tempo
        
    def add_gear(self, name: str, freq: float, callback: Callable, rhythmic: bool = False):
        gear = SynchronizedGear(name, freq, rhythmic=rhythmic)
        gear.callback = callback
        self.gears[name] = gear
        logger.info(f"⚙️ Gear '{name}' mounted with freq {freq}Hz (Rhythmic: {rhythmic})")

    def spin(self, override_dt: float = None):
        """Turns all registered gears."""
        now = time.time()
        # Use provided dt or calculate from real time
        dt = override_dt if override_dt is not None else (now - self.last_spin_time)
        self.last_spin_time = now
        
        # [V2.0] Get current Rhythm Gate value
        gate_value = self.rhythm.get_gate_value()

        for name, gear in self.gears.items():
            gear.rotate(dt)
            if gear.is_in_resonance(gate_value):
                if gear.callback and not gear._is_executing:
                    # [PHASE 250] Presence-based Execution (Pool-based)
                    gear._is_executing = True
                    def _task_wrapper(g):
                        try:
                            g.callback()
                        except Exception as e:
                            logger.error(f"Error in gear '{g.name}': {e}")
                        finally:
                            g._is_executing = False
                    
                    self.executor.submit(_task_wrapper, gear)

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
