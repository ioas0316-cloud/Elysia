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
        self.friction = 0.05      # Natural decay (Resistance)
        self.elasticity = 0.1     # Tendency to return to Z-axis

        # [Enhanced Geometric Properties]
        self.phase_lag = 0.0      # Resistance mapping: Delay in signal propagation (radians)
        self.capacitance = 0.0    # Phase Storage mapping: Ability to hold potential
        self.charge = 0.0         # Current stored potential energy

        # 4. Phase-Sync Coupling
        self.coupling_strength = 0.2
        self.interference_threshold = 0.1
        self.neighbors: List['RotorGate'] = []

    def set_electronic_attributes(self, resistance: float = 0.0, capacitance: float = 0.0):
        """Maps electronic circuit values to Rotor properties."""
        self.friction = 0.05 + (resistance * 0.1)
        self.phase_lag = resistance * 0.5 # Delay in radians
        self.capacitance = capacitance

    def connect(self, other: 'RotorGate'):
        if other not in self.neighbors:
            self.neighbors.append(other)
            other.neighbors.append(self)

    def process_stimulus(self, intensity: float, phase: float, dt: float):
        """
        Structural Computing Logic with Phase Lag and Storage:
        - Intensity: Tilts the axis away from Z.
        - Phase: Accelerates/Decelerates based on alignment (delayed by phase_lag).
        - Capacitance: Stores phase energy.
        """
        # Apply Phase Lag (Resistance effect)
        effective_phase = phase - self.phase_lag

        # 1. Axis Tilting (Intensity maps to X-Y deflection)
        tilt_force = np.array([intensity, 0.0, -intensity])
        self.axis = (self.axis + tilt_force * dt).astype(float)
        norm = np.linalg.norm(self.axis)
        if norm > 0: self.axis /= norm

        # 2. Phase Storage (Capacitance effect)
        if self.capacitance > 0:
            charge_push = intensity * dt
            self.charge = min(self.capacitance, self.charge + charge_push)
            # Use stored charge to amplify stimulus
            effective_intensity = intensity + (self.charge / self.capacitance) * 0.5
        else:
            effective_intensity = intensity

        # 3. Velocity Update (Resonance)
        alignment = math.cos(effective_phase - self.angle)
        acceleration = (alignment * effective_intensity) / self.mass
        self.velocity += acceleration * dt

        # 4. Natural Decay & Discharge
        self.velocity *= (1.0 - self.friction * dt)
        if self.capacitance > 0:
            discharge = (self.charge * 0.1) * dt
            self.charge -= discharge
            self.velocity += (discharge * 5.0) # Convert charge to momentum

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
            "active_intensity": math.sqrt(self.axis[0]**2 + self.axis[1]**2),
            "charge": self.charge
        }

class InterferenceGate(RotorGate):
    """
    A specialized RotorGate that acts as a logical gate.
    The state is determined by the interference pattern of multiple inputs.
    """
    def __init__(self, gate_id: str, gate_type: str = "AND"):
        super().__init__(gate_id)
        self.gate_type = gate_type
        self.inputs: List[Dict[str, float]] = [] # List of (intensity, phase)

    def add_input_signal(self, intensity: float, phase: float):
        self.inputs.append({"intensity": intensity, "phase": phase})

    def process_vortex_logic(self, dt: float):
        """
        Logic determined by wave interference rather than binary gates.
        """
        if not self.inputs:
            return

        # Calculate interference pattern
        total_x = sum(i["intensity"] * math.cos(i["phase"]) for i in self.inputs)
        total_y = sum(i["intensity"] * math.sin(i["phase"]) for i in self.inputs)

        result_intensity = math.sqrt(total_x**2 + total_y**2)
        result_phase = math.atan2(total_y, total_x)

        # Apply the resultant wave to the rotor
        self.process_stimulus(result_intensity, result_phase, dt)

        # Clear inputs for next cycle
        self.inputs = []

if __name__ == "__main__":
    # Test a single gate
    gate = RotorGate("TestGate")
    print(f"Initial: {gate.exhale()}")

    # Stimulate with high intensity and specific phase
    for _ in range(100):
        gate.process_stimulus(intensity=0.8, phase=math.pi/2, dt=0.1)
        gate.update(0.1)

    print(f"Final: {gate.exhale()}")
