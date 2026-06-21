import time
import numpy as np
from typing import Callable

class PCRVirtualScheduler:
    """
    [Synaptic Architecture] Virtual PCR Scheduler
    Temperature (T) is the master axis controlling:
    1. Operational Frequency (f)
    2. Spatial Resolution (σ)
    3. Stochastic Jitter (η)
    """
    def __init__(self, base_freq: float = 10.0, base_res: int = 256):
        self.temperature = 1.0
        self.base_freq = base_freq
        self.base_res = base_res

    def set_temperature(self, t: float):
        self.temperature = max(0.01, t)

    def get_physical_params(self):
        """
        Derive physical constraints from the thermal axis.
        """
        # Frequency (f): High T = High Frequency micro-vibrations
        freq = self.base_freq * (self.temperature ** 0.5)
        dt = 1.0 / freq

        # Spatial Resolution (σ): High T = Microscopic (High Res / Low Sigma Blur)
        # Low T = Macroscopic (Low Res / High Sigma Blur)
        # Note: In this simulation, resolution is handled as a sampling density
        sampling_density = int(self.base_res * (self.temperature ** 0.2))

        # Stochastic Jitter (η): High T = High Entropic Noise
        jitter = 0.1 * self.temperature

        return {
            "dt": dt,
            "frequency": freq,
            "sampling_density": sampling_density,
            "jitter": jitter,
            "temperature": self.temperature
        }

    def cognitive_loop(self, duration: float, step_func: Callable[[float, dict], None]):
        """
        Execute the loop governed by the thermal physical parameters.
        """
        start_time = time.time()
        elapsed = 0

        print(f"[PCR Scheduler] Loop started. Initial T={self.temperature:.2f}")

        while elapsed < duration:
            params = self.get_physical_params()

            # Perform the step within the physical constraints
            step_func(elapsed, params)

            # Physical delay (scaled for simulation observation)
            time.sleep(params["dt"] * 0.05)

            elapsed = time.time() - start_time

if __name__ == "__main__":
    scheduler = PCRVirtualScheduler()
    def example_step(t, p):
        print(f"[t={t:.2f}] T={p['temperature']:.2f}, F={p['frequency']:.1f}Hz, Res={p['sampling_density']}")

    scheduler.set_temperature(3.0)
    scheduler.cognitive_loop(0.5, example_step)

    scheduler.set_temperature(0.1)
    scheduler.cognitive_loop(0.5, example_step)
