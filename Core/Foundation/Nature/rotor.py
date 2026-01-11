"""
Rotor (자전축 로터)
==================================

"The Rotor is the Oscillator."
"로터가 곧 진동자다."

This module defines the Physical Rotor that powers the Hyper-Cosmos.
Updated to hold **7D WaveDNA**.

Structure:
- Base RPM: The carrier wave.
- DNA: The 7D genetic code (Physical, Functional, Phenomenal, etc.)
"""

from dataclasses import dataclass, field
import math
import random
from Core.Foundation.Wave.wave_dna import WaveDNA

@dataclass
class RotorConfig:
    """로터 설정"""
    rpm: float = 0.0          # Active Target RPM
    idle_rpm: float = 60.0    # Sleep/Breathing RPM
    mass: float = 1.0         # Amplitude proxy
    acceleration: float = 100.0

class Rotor:
    def __init__(self, name: str, config: RotorConfig, dna: WaveDNA = None):
        self.name = name
        self.config = config

        # [NEW] 7D DNA
        self.dna = dna if dna else WaveDNA(label=name)
        if dna:
            self.dna.normalize()

        # Dynamic State
        self.current_angle = random.uniform(0, 360)
        self.current_rpm = config.idle_rpm
        self.target_rpm = config.idle_rpm
        self.is_spinning = True
        
        # Energy State
        self.energy = 0.5 # 0.0 ~ 1.0

    @property
    def frequency_hz(self) -> float:
        return self.current_rpm / 60.0

    def wake(self, intensity: float = 1.0):
        """Excites the rotor."""
        self.target_rpm = self.config.rpm * intensity
        self.energy = min(1.0, self.energy + intensity * 0.5)

    def relax(self):
        """Returns to idle."""
        self.target_rpm = self.config.idle_rpm
        self.energy *= 0.99 # Decay

    def update(self, dt: float):
        """Physics Step."""
        # RPM Interpolation
        if self.current_rpm != self.target_rpm:
            diff = self.target_rpm - self.current_rpm
            change = self.config.acceleration * dt
            if abs(diff) < change:
                self.current_rpm = self.target_rpm
            else:
                self.current_rpm += change * (1 if diff > 0 else -1)

        # Angle Update
        if self.current_rpm > 0:
            degrees = (self.current_rpm / 60.0) * 360.0 * dt
            self.current_angle = (self.current_angle + degrees) % 360.0
            
        # Energy Decay
        if self.target_rpm == self.config.idle_rpm:
            self.energy *= (1.0 - (0.1 * dt)) # Natural decay

    def __repr__(self):
        return f"Rotor({self.name} | {self.current_rpm:.1f} RPM | E:{self.energy:.2f})"
