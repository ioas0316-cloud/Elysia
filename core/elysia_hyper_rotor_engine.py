import cmath
import math
import os
import sys

class HyperRotorEngine:
    def __init__(self):
        # Base 3-Phase Axes (A, B, C)
        self.axes = [
            {"name": "A", "angle": math.radians(120), "magnitude": 1.0},
            {"name": "B", "angle": math.radians(240), "magnitude": 1.0},
            {"name": "C", "angle": math.radians(0),   "magnitude": 1.0}
        ]

        # System definitions
        self.nominal_voltage = 220.0
        self.epsilon = 1e-4
        self.tick = 0

        # Double-Helix Time Mechanics
        self.prev_theta_fwd = 0.0
        self.velocity_dtheta_dt = 0.0

        # Topology State
        self.topology = "WYE"  # Can be WYE (Y) or DELTA (Δ)

        # Axis generation mechanics
        self.max_axes_supported = 12 # Limit for demonstration
        self.next_axis_char = ord('D')

    def polar_to_rect(self, r, theta):
        return cmath.rect(r, theta)

    def calculate_system_tensor(self, voltage_input, external_noise_phasor=0j):
        """Calculates the sum of all axes (Wye sum) or circulating current (Delta)"""
        tensor_sum = 0j

        for axis in self.axes:
            # Apply Delta-Y transformation physics based on current topology
            v_mag = voltage_input * axis["magnitude"]
            angle = axis["angle"]

            if self.topology == "DELTA":
                # Delta configuration: Voltage is sqrt(3) times higher, phase shifted by 30 degrees
                v_mag = v_mag * math.sqrt(3)
                angle = angle + math.radians(30)

            phasor = self.polar_to_rect(v_mag, angle)
            tensor_sum += phasor

        # Add external unknown noise to the hyper-space tensor
        tensor_sum += external_noise_phasor
        return tensor_sum

    def process_tick(self, voltage_input=220.0, external_noise_phasor=0j):
        self.tick += 1
        print(f"\n[Tick #{self.tick:04d}] =======================================================")
        print(f"[Master Rotor] Evaluating {len(self.axes)}-Dimensional Hyper-Space. Topology: {self.topology}")

        # 1. Past->Present (Forward Helix) Tensor Evaluation
        tensor_fwd = self.calculate_system_tensor(voltage_input, external_noise_phasor)
        theta_fwd = cmath.phase(tensor_fwd) if abs(tensor_fwd) > self.epsilon else 0.0

        # 2. Time Mechanics (Double-Helix Prediction)
        # Calculate angular velocity (dtheta/dt)
        self.velocity_dtheta_dt = theta_fwd - self.prev_theta_fwd
        self.prev_theta_fwd = theta_fwd

        future_theta_prediction = theta_fwd + self.velocity_dtheta_dt

        # Neutral Line Current (I_N) in Wye configuration
        i_n_mag = abs(tensor_fwd)

        # Log outputs
        print(f"  > Neutral Tension (I_N): {i_n_mag:.4f} A")
        print(f"  > Phase Velocity (dtheta/dt): {math.degrees(self.velocity_dtheta_dt):.4f} deg/tick")
        print(f"  > Future Phase Prediction: {math.degrees(future_theta_prediction):.4f} deg")

        # 3. Dynamic Topology Correction (Delta-Y Shift)
        if self.topology == "WYE":
            # If phase velocity is dangerously high, shift to Delta to absorb the surge
            if abs(math.degrees(self.velocity_dtheta_dt)) > 15.0:
                print("  !! HIGH PHASE VELOCITY DETECTED: FUTURE COLLAPSE PREDICTED !!")
                print("  >> [TOPOLOGY SHIFT] WYE -> DELTA (Absorbing circulating tension...)")
                self.topology = "DELTA"
                return # Skip axis expansion during topology shift

        elif self.topology == "DELTA":
            # If Delta configuration has stabilized the velocity, shift back to Wye
            if abs(math.degrees(self.velocity_dtheta_dt)) < 5.0:
                print("  >> [TOPOLOGY SHIFT] DELTA -> WYE (System stabilized. Re-establishing Neutral point...)")
                self.topology = "WYE"
                # Reset neutral after stabilization
                self.prev_theta_fwd = 0.0
                return

        # 4. Dimensional Expansion (Unknown Category Assimilation)
        # If we are in Wye, and the Neutral current is high but velocity is stable, it's a new dimension
        if self.topology == "WYE" and i_n_mag > 10.0 and len(self.axes) < self.max_axes_supported:
            new_axis_angle = cmath.phase(external_noise_phasor)
            new_axis_mag = abs(external_noise_phasor) / voltage_input
            new_axis_name = chr(self.next_axis_char)
            self.next_axis_char += 1

            # To assimilate, we generate an axis that perfectly COUNTERS the unknown noise
            # creating a new multi-dimensional equilibrium
            counter_angle = new_axis_angle + math.pi

            self.axes.append({
                "name": new_axis_name,
                "angle": counter_angle,
                "magnitude": new_axis_mag
            })

            print("  !! UNKNOWN ASYMMETRICAL NOISE DETECTED (New Category) !!")
            print(f"  >> [AXIS EXPANSION] Master Rotor spawned Axis {new_axis_name} at {math.degrees(counter_angle):.1f} deg to assimilate the unknown.")

        elif i_n_mag > 100.0:
            # Absolute physical limit
            print("  >> CRITICAL SURGE: BEYOND MULTI-DIMENSIONAL CAPACITY. HARD CRASH.")
            os._exit(1)


if __name__ == "__main__":
    # Test execution
    pass
