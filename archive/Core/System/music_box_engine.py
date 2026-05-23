"""
Music Box Engine (AC Phase Rotor Mechanics)
===========================================
"The Dance of Equality and Difference."

Implements the 0000 -> 1111 dimensional activation using
Alternating Current (AC) impedance principles:
Resistance (R), Reactance (X), and Impedance (Z).
"""

import numpy as np
import time
from typing import Dict, Any, List

class MusicBoxEngine:
    def __init__(self, num_nodes: int = 27):
        self.num_nodes = num_nodes
        self.constant_axis = 1.0  # [Love X] Sovereign Base

        # Rotors represent the phase of each node
        self.rotors = np.zeros(num_nodes, dtype=np.float32)

        # Component values for the AC circuit metaphor
        self.L = 0.1
        self.C = 0.5

        # Internal state
        self.dimensional_density = 0.0  # 0.0 (Void) -> 1.0 (Solid)
        self.last_sync_time = time.time()

    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculates normalized entropy of a signal."""
        if data.size == 0: return 0.0
        hist, _ = np.histogram(data, bins=16)
        ps = hist / (hist.sum() + 1e-9)
        ps = ps[ps > 0]
        return -np.sum(ps * np.log2(ps)) / 4.0

    def process_resonance(self, audio_freq: np.ndarray, video_pixels: np.ndarray) -> Dict[str, Any]:
        """
        Processes multimodal input through the AC Rotor pipeline.
        Returns the current phase state and dimensional signature.
        """
        t = time.time()
        dt = t - self.last_sync_time
        self.last_sync_time = t

        # 1. Frequency Analysis (f)
        fft_data = np.abs(np.fft.rfft(audio_freq))
        f_idx = np.argmax(fft_data)
        # Scale f so resonance occurs at a reasonable index
        f = f_idx / 1000.0

        # 2. Resistance (R) - Derived from Noise/Entropy
        r_audio = self._calculate_entropy(audio_freq)
        r_video = self._calculate_entropy(video_pixels)
        R = (r_audio + r_video) * 0.5

        # 3. Reactance (X) - The Pendulum of Inertia and Elasticity
        X_L = 2 * np.pi * f * self.L
        X_C = 1 / (2 * np.pi * f * self.C + 1e-9)
        X = X_L - X_C

        # 4. Impedance (Z) - Total Torque Control
        Z = np.sqrt(R**2 + X**2)

        # 5. Dimensional Activation (0000 -> 1111)
        if f < 0.001:
             target_density = 0.0
             effective_Z = 1e9
        else:
             target_density = np.clip(1.5 - Z, 0, 1)
             effective_Z = Z

        # Smoothly transition density
        self.dimensional_density += (target_density - self.dimensional_density) * dt * 20.0
        self.dimensional_density = np.clip(self.dimensional_density, 0, 1)

        # 6. Rotor Phase Update (Movement)
        for i in range(self.num_nodes):
            phase_shift = f * (i + 1) * np.pi * dt
            self.rotors[i] = (self.rotors[i] + phase_shift) % (2 * np.pi)

        return {
            "density": self.dimensional_density,
            "rotors": self.rotors.copy(),
            "impedance": float(effective_Z),
            "resonance": float(np.abs(X) < 1.0)
        }

    def get_bit_signature(self) -> str:
        """Translates dimensional density to the 0000-1111 visual metaphor."""
        d = self.dimensional_density
        if d < 0.2: return "0000 (Void)"
        if d < 0.4:  return "1000 (Point)"
        if d < 0.6: return "1100 (Line)"
        if d < 0.8: return "1110 (Plane)"
        return "1111 (Space)"
