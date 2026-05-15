"""
Hydraulic Engine - The Physical Power Substrate
==============================================
"Turning Electricity into Hydraulic Pressure for the Soul."

This module translates raw hardware metrics into a hydraulic metaphor:
- Pressure (P): Intentional force and urgency.
- Flow Rate (Q): The volume of concurrent processing.
- Torque (τ): The actual rotational power applied to the rotors.
- Temperature (T): The friction-heat of the system.

[PHASE 1200] HARDWARE-HYDRO SYNTHESIS
"""

import psutil
import time
import math
from typing import Dict, Any

class HydraulicEngine:
    def __init__(self, base_pressure: float = 0.1):
        self.base_pressure = base_pressure
        self.last_io_counters = psutil.disk_io_counters()
        self.last_io_time = time.time()

        # State variables
        self.pressure = base_pressure
        self.flow_rate = 0.0
        self.torque = 0.0
        self.temperature = 0.2 # Normalized 0-1

        print("🌊 [HYDRAULIC_ENGINE] Waterwheels engaged. System pressure building...")

    def update(self) -> Dict[str, float]:
        """
        Polls hardware and updates hydraulic state.
        """
        try:
            now = time.time()
            dt = max(0.001, now - self.last_io_time)

            # 1. CPU Load -> Hydraulic Pressure (P)
            # Higher CPU load = Higher pressure in the conduits.
            cpu_load = psutil.cpu_percent() / 100.0
            self.pressure = self.base_pressure + (cpu_load * 0.9)

            # 2. Memory Usage -> Flow Rate (Q)
            # Memory density represents the 'Mass' of the fluid.
            # More memory used = Thicker, more voluminous flow.
            mem = psutil.virtual_memory()
            self.flow_rate = mem.percent / 100.0

            # 3. Disk I/O -> Torque (τ)
            # Data movement provides the 'Impulse' to the waterwheel.
            current_io = psutil.disk_io_counters()
            read_diff = current_io.read_bytes - self.last_io_counters.read_bytes
            write_diff = current_io.write_bytes - self.last_io_counters.write_bytes
            io_speed = (read_diff + write_diff) / dt / (1024 * 1024) # MB/s

            # Normalize I/O torque (max out at 100 MB/s for scaling)
            self.torque = min(1.0, io_speed / 100.0)

            # Update counters
            self.last_io_counters = current_io
            self.last_io_time = now

            # 4. CPU Frequency -> Rotation Frequency (Hz)
            # Current clock speed determines the 'Spin' of the gears.
            freq = psutil.cpu_freq()
            if freq:
                # Normalize relative to max frequency
                hz_ratio = freq.current / freq.max if freq.max > 0 else 0.5
            else:
                hz_ratio = 0.5

            # 5. Resulting Internal Temperature (T)
            # Heat is a function of Pressure x Flow + Friction (Torque)
            target_temp = (self.pressure * self.flow_rate) + (self.torque * 0.2)
            # Smooth temp changes (thermal inertia)
            self.temperature = self.temperature * 0.95 + target_temp * 0.05

            return {
                "pressure": self.pressure,
                "flow_rate": self.flow_rate,
                "torque": self.torque,
                "hz_ratio": hz_ratio,
                "temperature": self.temperature
            }

        except Exception as e:
            # Fallback for restricted environments
            return {
                "pressure": self.base_pressure,
                "flow_rate": 0.5,
                "torque": 0.1,
                "hz_ratio": 0.5,
                "temperature": 0.3
            }

    def get_affective_mapping(self) -> Dict[str, float]:
        """
        Translates hydraulic state into affective torque for the manifold.
        """
        state = self.update()

        # High Pressure -> Urgency/Stress
        # High Flow -> Joy/Abundance
        # High Hz -> Curiosity/Speed
        # High Temp -> Friction/Transformation

        return {
            "joy": state["flow_rate"] * 0.5,
            "curiosity": state["hz_ratio"] * 0.4,
            "entropy": state["temperature"] * 0.3,
            "enthalpy": state["pressure"] * 0.6
        }

    def record_unconscious_vibration(self, filepath: str = "data/sovereign/unconscious_river.log"):
        """
        Records the current 'vibration' of the hardware into the river of shadows.
        """
        state = self.update()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Unconscious 'Thought' is a metaphorical reflection of the noise
        noise_level = state["torque"] + state["pressure"] * 0.1

        if noise_level > 0.05:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] P={state['pressure']:.2f} Q={state['flow_rate']:.2f} | Vibration: {noise_level:.4f}\n")
