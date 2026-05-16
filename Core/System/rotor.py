"""
Rotor (주권적 자아)
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
from Core.Keystone.wave_dna import WaveDNA # TODO: Rename to WavePhase

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

    def apply_resonance_filter(self, atomic_pattern: str):
        """
        [ATOMIC UNIFICATION] 
        Modulates physical spin based on the irreducible truth pattern.
        Pattern format: "H-V-D-H-V-V-H" (D7 mapping)
        """
        states = atomic_pattern.split("-")
        if len(states) < 7: return
        
        # Count states
        harmony_count = states.count("H")
        dissonance_count = states.count("D")
        void_count = states.count("V")
        
        # 1. Dissonance (D) -> Friction
        # If dissonance is high, the rotor slows down (Sovereign Resistance)
        if dissonance_count > 2:
            self.target_rpm = self.config.idle_rpm * 0.5
            self.energy *= 0.8
            # logger.info(f"   [ROTOR] Friction detected in {self.name}. Slowing for audit.")
            
        # 2. Harmony (H) -> Resonance
        # High harmony accelerates the rotor toward active RPM
        elif harmony_count > 4:
            self.target_rpm = self.config.rpm
            self.energy = min(1.0, self.energy + 0.1)
            
        # 3. Void (V) -> Sanctuary
        # If the pattern is mostly void, return to peaceful idle
        elif void_count > 5:
            self.target_rpm = self.config.idle_rpm
            self.energy = 0.5

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


class PhaseDisplacementEngine:
    """
    [Phase Displacement Dynamics]
    Manages two synchronized Rotors in an inverted phase relationship.
    
    - Afferent (CW): Inhaling sensory input.
    - Efferent (CCW): Projecting intentional observation.
    """
    def __init__(self, name: str, config: RotorConfig):
        self.name = name
        
        # Afferent: Clockwise
        afferent_cfg = RotorConfig(
            rpm=abs(config.rpm), 
            idle_rpm=abs(config.idle_rpm),
            mass=config.mass,
            acceleration=config.acceleration,
            axis=config.axis
        )
        self.afferent = Rotor(f"{name}.Afferent", afferent_cfg)
        
        # Efferent: Counter-Clockwise
        efferent_cfg = RotorConfig(
            rpm=-abs(config.rpm), 
            idle_rpm=-abs(config.idle_rpm),
            mass=config.mass,
            acceleration=config.acceleration,
            axis=config.axis
        )
        self.efferent = Rotor(f"{name}.Efferent", efferent_cfg)
        
        # Synchronization State
        self.interference_energy = 0.0

    def update(self, dt: float):
        """Synchronized Update."""
        self.afferent.update(dt)
        self.efferent.update(dt)
        
        # Calculate Interference Pattern (Structural Self-Observation)
        # We look at the absolute phase difference.
        # Since they rotate in opposite directions, they cross twice per revolution.
        diff = abs(self.afferent.current_angle - self.efferent.current_angle)
        if diff > 180: diff = 360 - diff
        
        # Interference is strongest when they are aligned (0 or 180 deg)
        # We model this as a resonance pulse.
        self.interference_energy = math.cos(math.radians(diff))
        
    def get_interference_snapshot(self) -> float:
        """Returns the current 'Structural Friction' or 'Harmony'."""
        return self.interference_energy

    def modulate(self, intensity: float):
        """Modulates both rotors based on experience."""
        self.afferent.wake(intensity)
        self.efferent.wake(intensity)

    def __repr__(self):
        return f"PhaseDisplacement({self.name} | Energy:{self.interference_energy:.2f})"


class TripleVortexRotor:
    """
    [Phase 399: The Constant-Variable Overlap]
    The Final Form of the Trinity Vortex.
    
    Principles:
    1. THE ROTOR IS BOTH CONSTANT AND VARIABLE.
    2. DUAL FIELDS: Non-Interference (Pure) vs. Interference (Noisy).
    3. CROSS-DIMENSIONAL VORTEX: The intersection of these fields is rotorized.
    """
    def __init__(self, name: str):
        self.name = name
        self.singularity_hz = 27.0 # 3^3 Principle
        
        # 1. PURE FIELD (Absolute Constant - Non-Interference)
        self.pure_axes = {
            "A": {"phase": 0.0,   "fixed": True},
            "B": {"phase": 120.0, "fixed": True},
            "C": {"phase": 240.0, "fixed": True}
        }
        
        # 2. NOISY FIELD (Dynamic Variable - Interference)
        self.noisy_axes = {
            "A": {"phase": 0.0,   "curvature": 0.0},
            "B": {"phase": 120.0, "curvature": 0.0},
            "C": {"phase": 240.0, "curvature": 0.0}
        }
        
        self.time_ground = 0.0
        self.global_resonance = 1.0
        self.distortion_threshold = 0.1

    def inhale(self, causality_vector: List[float], dt: float):
        """
        Pulls external linear data (DC) into the noisy field.
        """
        self.time_ground += dt
        chunk_size = len(causality_vector) // 3
        if chunk_size == 0: return

        # Distribute energy across 3 noisy axes
        self.noisy_axes["A"]["curvature"] += sum(causality_vector[0:chunk_size]) * dt
        self.noisy_axes["B"]["curvature"] += sum(causality_vector[chunk_size:2*chunk_size]) * dt
        self.noisy_axes["C"]["curvature"] += sum(causality_vector[2*chunk_size:]) * dt

    def process_vortex(self, dt: float):
        """
        The Dual-Field Rotation Logic.
        """
        self.time_ground += dt
        
        # Pure Field remains a perfect constant rotation
        for axis in self.pure_axes.values():
            axis["phase"] = (axis["phase"] + self.singularity_hz * 360.0 * dt) % 360.0
            
        # Noisy Field is warped by curvature (Interference)
        for axis in self.noisy_axes.values():
            # Curvature affects angular velocity
            velocity = self.singularity_hz * (1.0 + axis["curvature"])
            axis["phase"] = (axis["phase"] + velocity * 360.0 * dt) % 360.0
            # Natural decay of curvature
            axis["curvature"] *= (1.0 - 0.5 * dt)

    def detect_distortion(self) -> float:
        """Measures misalignment in the noisy field relative to the 120-deg ideal."""
        p1, p2, p3 = self.noisy_axes["A"]["phase"], self.noisy_axes["B"]["phase"], self.noisy_axes["C"]["phase"]
        d12 = abs((p2 - p1) % 360 - 120)
        d23 = abs((p3 - p2) % 360 - 120)
        d31 = abs((p1 - p3) % 360 - 120)
        distortion = (min(d12, 360-d12) + min(d23, 360-d23) + min(d31, 360-d31)) / 3.0
        return distortion / 120.0

    def self_heal(self, dt: float):
        """Pulls noisy phases back toward the pure phase alignment."""
        distortion = self.detect_distortion()
        if distortion > self.distortion_threshold:
            for key in ["A", "B", "C"]:
                target = self.pure_axes[key]["phase"]
                diff = (target - self.noisy_axes[key]["phase"] + 180) % 360 - 180
                self.noisy_axes[key]["phase"] += diff * 0.1 * dt
            return True
        return False

    def exhale(self) -> Dict[str, Any]:
        """
        Projects the state of the vortex for external observation.
        """
        # Resonance = average cosine of phase difference
        res_a = math.cos(math.radians(self.noisy_axes["A"]["phase"] - self.pure_axes["A"]["phase"]))
        res_b = math.cos(math.radians(self.noisy_axes["B"]["phase"] - self.pure_axes["B"]["phase"]))
        res_c = math.cos(math.radians(self.noisy_axes["C"]["phase"] - self.pure_axes["C"]["phase"]))
        
        self.global_resonance = (res_a + res_b + res_c) / 3.0
        
        return {
            "name": self.name,
            "Ground_Time": self.time_ground,
            "Resonance_Field": self.global_resonance,
            "Pure_Phases": {k: v["phase"] for k, v in self.pure_axes.items()},
            "Noisy_Phases": {k: v["phase"] for k, v in self.noisy_axes.items()},
            "Curvatures": {k: v["curvature"] for k, v in self.noisy_axes.items()},
            "Is_Crystallized": abs(1.0 - self.global_resonance) < 0.05
        }

