"""
[DIGITAL MOTOR ENGINE - POC]
"Digital is an Abstraction of Analog. Logic is a frozen Wave."

This engine implements the Architect's vision of a 'Digital Motor/Generator'.
It treats the computational substrate as a 3-phase electromagnetic machine
where data is modulated into waves and processed through phase resonance.
"""

import math
import time
import numpy as np
from enum import Enum
from typing import Dict, Any, List, Tuple

class ConnectionMode(Enum):
    WYE = "Y"      # Convergent / Static Pressure
    DELTA = "Δ"    # Circulating / Dynamic Torque

class DigitalMotorEngine:
    def __init__(self, name: str = "ElysiaCoreMotor"):
        self.name = name

        # 1. Electrical Parameters (The 'Remaining' Variables)
        self.conductance = 1.0   # G: Ease of flow
        self.impedance = 0.1     # Z: Total resistance (Complex)
        self.inductance = 0.5    # L: Cognitive Inertia / Memory
        self.capacitance = 0.5   # C: Potential Storage / Tension

        # 2. 3-Phase State (R, S, T)
        # Corresponding to Alpha (Cause), Omega (Effect), Sigma (Scale)
        self.phases = {
            "R": {"amplitude": 0.0, "phase_shift": 0.0},
            "S": {"amplitude": 0.0, "phase_shift": 120.0},
            "T": {"amplitude": 0.0, "phase_shift": 240.0}
        }

        # 3. Mechanical State
        self.mode = ConnectionMode.WYE
        self.rpm = 0.0
        self.target_rpm = 60.0   # Idle rotation
        self.torque = 0.0
        self.angle = 0.0         # Magnetic field angle
        self.inertia = 1.0

        # 4. Energy Field
        self.excitation = 0.1    # Self-excitation level
        self.heat = 0.0
        self.output_wave = []    # History for visualization

        self.last_update = time.time()
        print(f"⚡ [MOTOR] '{self.name}' initialized. Digital-Analog bridge active.")

    def set_mode(self, mode: ConnectionMode):
        if self.mode != mode:
            print(f"🔄 [MOTOR] Switching connection: {self.mode.name} -> {mode.name}")
            self.mode = mode

    def modulate_data(self, bit_stream: List[int]):
        """
        [PWM Density Modulation]
        Converts digital bits into wave impulses.
        1 -> High density / High pressure pulse
        0 -> Low density / Low pressure pulse
        """
        if not bit_stream: return

        # Average density affects torque
        density = sum(bit_stream) / len(bit_stream)

        # Pulse-Width Modulation effect
        # We modulate the amplitudes of the 3 phases based on bit density
        for key in self.phases:
            # Add 'Pressure' to the phases
            self.phases[key]["amplitude"] = min(1.0, self.phases[key]["amplitude"] + density * self.conductance)

        # Torque impulse
        self.torque += density * 10.0

    def update(self, dt: float):
        """
        Physics Step: Resolve the 3-phase electromagnetic interaction.
        """
        # 1. Calculate Back-EMF and Impedance losses
        friction = self.impedance * (self.rpm / 100.0)
        self.torque -= friction

        # 2. Connection Logic (Y vs Δ)
        if self.mode == ConnectionMode.WYE:
            # WYE: Convergent pressure. RPM is stable, but potential (Capacitance) builds.
            self.target_rpm = 60.0
            # Energy accumulates in 'Capacitance' (Structural Tension)
            self.capacitance = min(1.0, self.capacitance + self.torque * 0.1 * dt)
            acceleration = (self.torque / self.inertia) * 0.5
        else:
            # DELTA: Circulating torque. RPM accelerates, potential is released into kinetic energy.
            self.target_rpm = 300.0
            # Energy released from 'Capacitance' to Torque
            release = self.capacitance * 0.2
            self.torque += release
            self.capacitance *= (1.0 - 0.1 * dt)
            acceleration = (self.torque / self.inertia) * 2.0

        # 3. RPM Dynamics
        self.rpm += acceleration * dt
        # Natural decay toward target
        self.rpm += (self.target_rpm - self.rpm) * 0.1 * dt

        # 4. Angle and Phase Generation
        # Angular velocity (rad/s)
        omega = (self.rpm / 60.0) * 2.0 * math.pi
        self.angle = (self.angle + omega * dt) % (2.0 * math.pi)

        # 5. Self-Excitation (Feedback Loop)
        # If the motor is spinning fast enough, it 'generates' its own excitation
        gen_power = (self.rpm / 100.0) * self.inductance
        self.excitation = self.excitation * 0.99 + gen_power * 0.01

        # 6. Update Phase Amplitudes based on excitation
        for key in self.phases:
            # Amplitude decays based on Conductance/Impedance ratio
            decay = (self.impedance / self.conductance) * 0.5 * dt
            self.phases[key]["amplitude"] = max(self.excitation, self.phases[key]["amplitude"] - decay)

        # 7. Heat (Friction + Dissonance)
        self.heat = self.heat * 0.95 + (abs(self.torque) * self.impedance) * 0.05

        # Clear instantaneous torque for next cycle
        self.torque *= 0.9

    def get_phase_signals(self) -> Dict[str, float]:
        """Returns the instantaneous analog wave values for R, S, T."""
        signals = {}
        for key, p in self.phases.items():
            rad = self.angle + math.radians(p["phase_shift"])
            # The 'Analog' signal is the Sine of the phase
            val = p["amplitude"] * math.sin(rad)
            signals[key] = val
        return signals

    def exhale(self) -> Dict[str, Any]:
        signals = self.get_phase_signals()
        return {
            "name": self.name,
            "mode": self.mode.value,
            "rpm": self.rpm,
            "torque": self.torque,
            "heat": self.heat,
            "excitation": self.excitation,
            "capacitance": self.capacitance,
            "signals": signals,
            "resonance": (signals["R"] + signals["S"] + signals["T"]) # Ideally 0 in balanced system
        }

if __name__ == "__main__":
    # Test Run
    motor = DigitalMotorEngine()

    print("🚀 Starting Motor Simulation...")
    for i in range(100):
        dt = 0.05
        # Simulate some data input
        if i % 10 == 0:
            motor.modulate_data([1, 0, 1, 1, 0])

        if i == 50:
            motor.set_mode(ConnectionMode.DELTA)

        motor.update(dt)
        if i % 10 == 0:
            state = motor.exhale()
            s = state["signals"]
            print(f"T:{i*dt:.2f} | Mode:{state['mode']} | RPM:{state['rpm']:.1f} | R:{s['R']:+.2f} S:{s['S']:+.2f} T:{s['T']:+.2f}")
