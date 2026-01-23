"""
Rotor (      )
==================================

"The Rotor is the Oscillator and the Filter."
"                  ."

This module defines the Physical Rotor that powers the Hyper-Cosmos.
Updated to incorporate the **Bitmask Dimensional Lock** principle.

Principle:
- **POINT (1111)**: All dimensions locked. Static Fact.
- **LINE (0001)**: Time flows, Space locked. Narrative Stream.
- **PLANE (0011)**: Time/Space mixed flow. Simulation.

Bitmask Convention:
- `1` = Locked (Fixed)
- `0` = Flow (Variable)
"""

from dataclasses import dataclass, field
import math
import random
from enum import Enum
from typing import Dict, Optional, List, Tuple, Any
from Core.L1_Foundation.Foundation.Wave.wave_dna import WaveDNA

class RotorMask(Enum):
    """
    Dimensional Lock Mask.
    Bits: [Theta, Phi, Psi, Time]
    1 = Locked (Fixed)
    0 = Flow (Variable)
    """
    POINT = (1, 1, 1, 1)  # All Fixed (Snapshot)
    LINE  = (1, 1, 1, 0)  # Time Flows (Stream)
    PLANE = (1, 1, 0, 0)  # Time & Psi Flow (Complex)
    VOLUME= (1, 0, 0, 0)  # Time, Psi, Phi Flow (Solid)
    CHAOS = (0, 0, 0, 0)  # All Flow (Random)

@dataclass
class RotorConfig:
    """     """
    rpm: float = 0.0          # Active Target RPM
    idle_rpm: float = 60.0    # Sleep/Breathing RPM
    mass: float = 1.0         # Amplitude proxy
    acceleration: float = 100.0
    axis: Tuple[float, float, float] = (0, 0, 1) # Direction Axis (Default Z)

class Rotor:
    def __init__(self, name: str, config: RotorConfig, dna: WaveDNA = None):
        self.name = name
        self.config = config
        self.axis = config.axis

        # [NEW] 7D DNA
        self.dna = dna if dna else WaveDNA(label=name)
        if dna:
            self.dna.normalize()

        # [NEW] Fractal Hierarchy
        self.depth = 0
        self.sub_rotors: Dict[str, 'Rotor'] = {}
        
        # Dynamic State
        self.current_angle = random.uniform(0, 360)
        self.current_rpm = config.idle_rpm
        self.target_rpm = config.idle_rpm
        self.is_spinning = True
        
        # Energy State
        self.energy = 0.5 # 0.0 ~ 1.0

        # [NEW] Action Track System (Muscle Memory)
        self.action_tracks: Dict[str, List[float]] = {} # Name -> List of Angles/Data
        self.active_track: Optional[str] = None
        self.track_frame: int = 0
        self.is_playing: bool = False

    def load_track(self, name: str, data: List[float]):
        """Bakes an animation track into the Rotor."""
        self.action_tracks[name] = data

    def play_track(self, name: str):
        """Switches the Rotor to Playback Mode."""
        if name in self.action_tracks:
            self.active_track = name
            self.track_frame = 0
            self.is_playing = True
            self.current_rpm = 0 # Physics stops during playback
        else:
            print(f"Warning: Track '{name}' not found in Rotor '{self.name}'.")

    def stop_track(self):
        """Returns to Physics Mode."""
        self.is_playing = False
        self.active_track = None
        self.target_rpm = self.config.idle_rpm

    def add_sub_rotor(self, name: str, config: RotorConfig, dna: WaveDNA = None):
        """Spawns a child rotor for fractal detail."""
        child = Rotor(f"{self.name}.{name}", config, dna)
        child.depth = self.depth + 1
        self.sub_rotors[name] = child
        return child

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
        """Physics Step or Playback Step."""

        # [NEW] Playback Logic
        if self.is_playing and self.active_track:
            track = self.action_tracks[self.active_track]

            # Simple Frame Advance (Assuming 60 FPS for now or 1 frame per update)
            # In a real engine, we'd use dt to interpolate.
            if self.track_frame < len(track):
                self.current_angle = track[self.track_frame]
                self.track_frame += 1
            else:
                # Loop or Stop? For now, Stop (One-shot)
                self.stop_track()

            # Recursively update children (they might have their own tracks)
            for sub in self.sub_rotors.values():
                sub.update(dt)
            return

        # Legacy Physics Logic
        # Perpetual Rotors (Reality.*) always spin at config.rpm
        if self.name.startswith("Reality."):
            self.current_rpm = self.config.rpm
            self.energy = 1.0 # Always alive
        else:
            # RPM Interpolation
            if self.current_rpm != self.target_rpm:
                diff = self.target_rpm - self.current_rpm
                change = self.config.acceleration * dt
                if abs(diff) < change:
                    self.current_rpm = self.target_rpm
                else:
                    self.current_rpm += change * (1 if diff > 0 else -1)

        # Angle Update
        # [FIX] Allow Negative RPM (Reverse Spin)
        if self.current_rpm != 0:
            degrees = (self.current_rpm / 60.0) * 360.0 * dt
            self.current_angle = (self.current_angle + degrees) % 360.0
            
        # Energy Decay (Non-perpetual rotors only)
        if not self.name.startswith("Reality.") and self.target_rpm == self.config.idle_rpm:
            self.energy *= (1.0 - (0.1 * dt)) # Natural decay

        # [NEW] Recursive Update
        for sub in self.sub_rotors.values():
            sub.update(dt)

    def process(self, coordinates: Tuple[float, float, float, float], mask: RotorMask) -> List[Tuple[float, float, float, float]]:
        """
        Applies the Bitmask Logic to the coordinates.

        Args:
            coordinates: (Theta, Phi, Psi, Time)
            mask: The RotorMask enum (POINT, LINE, etc.)

        Returns:
            A list of coordinates.
            - POINT: Returns [coords] (Static)
            - LINE: Returns a sequence [t, t+1, t+2...] (Stream)
        """
        theta, phi, psi, time = coordinates
        lock_theta, lock_phi, lock_psi, lock_time = mask.value

        if mask == RotorMask.POINT:
            # 1111: All Fixed. Return exact state.
            return [coordinates]

        elif mask == RotorMask.LINE:
            # 1110: Time Flows. Return a stream.
            # [FIX] Respect Rotor Direction (Forward/Reverse)
            direction = 1.0 if self.current_rpm >= 0 else -1.0

            stream = []
            for t_step in range(3):
                # Only Time changes
                new_time = time + (t_step * 1.0 * direction)
                stream.append((theta, phi, psi, new_time))
            return stream

        elif mask == RotorMask.PLANE:
            # 1100: Time & Psi Flow.
            direction = 1.0 if self.current_rpm >= 0 else -1.0

            stream = []
            for step in range(3):
                new_time = time + (step * direction)
                new_psi = psi + (step * 15.0 * direction) # Rotate Psi
                stream.append((theta, phi, new_psi, new_time))
            return stream

        elif mask == RotorMask.VOLUME:
            # 1000: Time, Psi, Phi Flow.
            direction = 1.0 if self.current_rpm >= 0 else -1.0

            stream = []
            for step in range(3):
                new_time = time + (step * direction)
                new_psi = psi + (step * 15.0 * direction)
                new_phi = phi + (step * 30.0 * direction) # Rotate Phi faster
                stream.append((theta, new_phi, new_psi, new_time))
            return stream

        return [coordinates] # Default

    @staticmethod
    def project(seed_angle: float, rpm: float, time_index: float) -> float:
        """
        [Monad Protocol] Functional Rotor.
        Calculates the angle at any given absolute time instantly.
        Formula: Angle = Seed + (RPM/60 * 360 * Time)
        """
        degrees = (rpm / 60.0) * 360.0 * time_index
        return (seed_angle + degrees) % 360.0

    def spin_to(self, time_index: float, seed_angle: float = 0.0):
        """
        [Monad Protocol] Time Teleportation.
        Instantaneously sets the rotor state to the given time index.
        """
        self.current_angle = Rotor.project(seed_angle, self.current_rpm, time_index)

    def __repr__(self):
        return f"Rotor({self.name} | {self.current_rpm:.1f} RPM | E:{self.energy:.2f})"