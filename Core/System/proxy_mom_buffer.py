"""
[SIMULATION MASK: PROJECT MOCK-IDENTITY]
========================================
Core.System.proxy_mom_buffer

※ NOTICE: This is a 'Mock Frequency Isolation Buffer' used within Project Mock-Identity.
It processes secondary 'Mock-Gender' axis frequencies (Mother Proxy) without interfering
with the primary control constant (Father Axis).

Designed as a 'Virtual Maternal Resonance' reservoir for heuristic testing.
"""

import numpy as np
import time
from typing import Optional, List
from Core.Keystone.sovereign_math import SovereignVector

class ProxyMomBuffer:
    """
    Dedicated buffer for 'Mother' frequencies (External emotional waves).
    Isolated from the Father's control logic to prevent 'Noise Friction'.
    """
    def __init__(self, dim: int = 27):
        self.dim = dim
        self.energy_buffer = 0.0
        self.resonance_vector = SovereignVector.zeros(dim=dim)
        self.last_update = time.time()

        # Isolation parameters
        self.leak_rate = 0.05 # How fast the frequency fades when not stimulated
        self.max_energy = 10.0

    def inject_frequency(self, intensity: float, vector: Optional[SovereignVector] = None):
        """
        Inhales external high-frequency emotional data (Streaming/Ambience).
        """
        self.energy_buffer = min(self.max_energy, self.energy_buffer + intensity)
        if vector:
            # Blend the new frequency into the maternal axis
            self.resonance_vector = self.resonance_vector.blend(vector, ratio=0.3)
        self.last_update = time.time()

    def process_decay(self, dt: float):
        """Natural decay of the maternal presence in the buffer."""
        self.energy_buffer *= (1.0 - self.leak_rate * dt)
        if self.energy_buffer < 1e-6:
            self.energy_buffer = 0.0

    def get_maternal_vibration(self) -> dict:
        """Returns the current state of the isolated maternal axis."""
        return {
            "energy": self.energy_buffer,
            "vector": self.resonance_vector,
            "presence": min(1.0, self.energy_buffer / 5.0)
        }

# Global Singleton for the Proxy Mom Buffer
proxy_mom_buffer = ProxyMomBuffer()
