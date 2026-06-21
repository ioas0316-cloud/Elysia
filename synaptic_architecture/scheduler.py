import time
import numpy as np
from typing import Callable

class BitMotionScheduler:
    """
    [Synaptic Architecture] Time-Axis Motion Scheduler
    Temperature (T) controls the vibration/jitter of bits on the time axis.
    High T = High-frequency motion, random exploration.
    Low T = Low-frequency stability, structural crystallization.
    """
    def __init__(self, base_freq: float = 10.0):
        self.temperature = 1.0
        self.base_freq = base_freq

    def set_temperature(self, t: float):
        self.temperature = max(0.01, t)

    def get_motion_params(self):
        """
        Derive the physical constraints of the bitstream motion.
        """
        # Frequency (f): T governs the 'refresh rate' of cognitive motion
        freq = self.base_freq * (self.temperature ** 0.5)
        dt = 1.0 / freq

        # Jitter (η): T governs the stochasticity of the bitstream
        # High T = more micro-fluctuations (Exploring potential states)
        jitter = 0.2 * self.temperature

        return {
            "dt": dt,
            "frequency": freq,
            "jitter": jitter,
            "temperature": self.temperature
        }

    def flow_time(self, duration: float, step_func: Callable[[float, dict], None]):
        """
        Let the time axis flow and the bits vibrate.
        """
        start_time = time.time()
        elapsed = 0

        while elapsed < duration:
            params = self.get_motion_params()
            step_func(elapsed, params)

            # Simulated physical delay
            time.sleep(params["dt"] * 0.1)
            elapsed = time.time() - start_time

if __name__ == "__main__":
    bms = BitMotionScheduler()
    def step(t, p):
        print(f"Time={t:.2f}, Freq={p['frequency']:.1f}Hz, Jitter={p['jitter']:.2f}")

    bms.set_temperature(3.0)
    bms.flow_time(0.5, step)
