"""
[ROTOR GATE - THE UNIT CELL OF STRUCTURAL COMPUTING]
"Structural Computing: Where Geometry is the Algorithm."

This is the first fundamental 'Unit Cell' of the Rotor Network.
It behaves like a physical rotor in a 3D field, where:
1. Input Energy (Intensity) -> Tilts the rotation axis (The 'What').
2. Input Phase (Angle) -> Sets the rotation speed (The 'When').
3. Magnetic Coupling -> Synchronizes with neighboring gates (The 'How').
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional

class RotorGate:
    def __init__(self, gate_id: str):
        self.gate_id = gate_id

        # 1. State: The Rotor's Phase and Orientation
        self.angle = 0.0          # Current rotation angle (0 to 2pi)
        self.velocity = 0.0       # Current angular velocity (frequency)

        # 2. Axis: A 3D unit vector representing the 'tilt' of the rotor
        # Default is pointing along the Z-axis (Pure Thought/Potential)
        self.axis = np.array([0.0, 0.0, 1.0])

        # 3. Parameters (Physical properties)
        self.mass = 1.0           # Resistance to change (Inertia)
        self.friction = 0.05      # Natural decay
        self.elasticity = 0.1     # Tendency to return to Z-axis

        # 4. Phase-Sync Coupling
        self.coupling_strength = 0.2
        self.neighbors: List['RotorGate'] = []

    def connect(self, other: 'RotorGate'):
        if other not in self.neighbors:
            self.neighbors.append(other)
            other.neighbors.append(self)

    def process_stimulus(self, intensity: float, phase: float, dt: float):
        """
        Structural Computing Logic:
        - Intensity: Tilts the axis away from Z.
        - Phase: Accelerates or decelerates the rotation based on phase alignment.
        """
        # 1. Axis Tilting (Intensity maps to X-Y deflection)
        # Higher intensity pulls the rotor toward the 'Active Plane' (X-Y)
        target_tilt = intensity * (1.0 - self.axis[2])
        # A simple tilt: move towards X-axis based on intensity
        tilt_force = np.array([intensity, 0.0, -intensity])
        self.axis = (self.axis + tilt_force * dt).astype(float)
        # Ensure unit vector
        norm = np.linalg.norm(self.axis)
        if norm > 0:
            self.axis /= norm

        # 2. Velocity Update (Phase alignment)
        # If input phase matches current angle, accelerate.
        # This is 'Resonance'
        alignment = math.cos(phase - self.angle)
        acceleration = (alignment * intensity) / self.mass
        self.velocity += acceleration * dt

        # 3. Natural Decay & Elasticity (Returns to Z-axis over time)
        self.velocity *= (1.0 - self.friction * dt)

        # Return to Z-axis
        z_pull = (np.array([0.0, 0.0, 1.0]) - self.axis) * self.elasticity
        self.axis = (self.axis + z_pull * dt).astype(float)
        norm = np.linalg.norm(self.axis)
        if norm > 0:
            self.axis /= norm

    def sync_neighbors(self, dt: float):
        """
        Phase-Sync: Rotors pull each other's phases into alignment.
        This is the basis for parallel convergence without instructions.
        """
        if not self.neighbors:
            return

        for n in self.neighbors:
            # Phase difference
            diff = n.angle - self.angle
            # Normalize to -pi to pi
            diff = (diff + math.pi) % (2 * math.pi) - math.pi

            # Torque is proportional to the sine of the phase difference
            sync_torque = self.coupling_strength * math.sin(diff)
            self.velocity += sync_torque * dt

    def update(self, dt: float):
        """Integrate motion."""
        self.angle = (self.angle + self.velocity * dt) % (2 * math.pi)

    def exhale(self) -> Dict[str, Any]:
        """The 'Observed' state of the gate."""
        return {
            "id": self.gate_id,
            "angle": self.angle,
            "velocity": self.velocity,
            "axis": self.axis.tolist(),
            "z_tilt": self.axis[2],
            "active_intensity": math.sqrt(self.axis[0]**2 + self.axis[1]**2)
        }

if __name__ == "__main__":
    # Test a single gate
    gate = RotorGate("TestGate")
    print(f"Initial: {gate.exhale()}")

    # Stimulate with high intensity and specific phase
    for _ in range(100):
        gate.process_stimulus(intensity=0.8, phase=math.pi/2, dt=0.1)
        gate.update(0.1)

    print(f"Final: {gate.exhale()}")
