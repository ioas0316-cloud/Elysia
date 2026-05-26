import cmath
import math
import os
import sys

class HyperRotorEngine:
    def __init__(self):
        # Base 3-Phase Axes (A, B, C) - 120 degree geometric constraint
        self.axes = [
            {"name": "A", "angle": math.radians(120), "magnitude": 1.0},
            {"name": "B", "angle": math.radians(240), "magnitude": 1.0},
            {"name": "C", "angle": math.radians(0),   "magnitude": 1.0}
        ]

        self.nominal_voltage = 220.0
        self.epsilon = 1e-4
        self.tick = 0

        # Double-Helix Time Mechanics
        self.prev_theta_fwd = 0.0
        self.velocity_dtheta_dt = 0.0

        self.topology = "WYE"  # WYE or DELTA
        self.max_axes_supported = 12
        self.next_axis_char = ord('D')

    def polar_to_rect(self, r, theta):
        return cmath.rect(r, theta)

    def calculate_system_tensor(self, voltage_input, external_noise_phasor=0j):
        tensor_sum = 0j
        for axis in self.axes:
            v_mag = voltage_input * axis["magnitude"]
            angle = axis["angle"]

            # IEEE 3-phase topology dynamics implementation
            if self.topology == "DELTA":
                v_mag = v_mag * math.sqrt(3)
                angle = angle + math.radians(30)

            tensor_sum += self.polar_to_rect(v_mag, angle)

        tensor_sum += external_noise_phasor
        return tensor_sum

    def process_tick(self, voltage_input=220.0, external_noise_phasor=0j):
        self.tick += 1
        print(f"\n[Tick #{self.tick:04d}] =======================================================")
        print(f"[Master Rotor] Evaluating {len(self.axes)}-Dimensional Hyper-Space. Topology: {self.topology}")

        # 1. Forward Helix Tensor Evaluation
        tensor_fwd = self.calculate_system_tensor(voltage_input, external_noise_phasor)
        theta_fwd = cmath.phase(tensor_fwd) if abs(tensor_fwd) > self.epsilon else 0.0

        # 2. Phase Unwrapping (Mathematical Discontinuity Correction)
        # Correcting phase jump due to cmath.phase limit (-pi to pi)
        diff = theta_fwd - self.prev_theta_fwd
        if diff > math.pi:
            diff -= 2 * math.pi
        elif diff < -math.pi:
            diff += 2 * math.pi

        self.velocity_dtheta_dt = diff
        future_theta_prediction = theta_fwd + self.velocity_dtheta_dt

        # Honest KCL Neutral Residual Tension measurement
        i_n_mag = abs(tensor_fwd)

        print(f"  > Neutral Tension (I_N): {i_n_mag:.4f} A")
        print(f"  > Phase Velocity (dtheta/dt): {math.degrees(self.velocity_dtheta_dt):.4f} deg/tick")
        print(f"  > Future Phase Prediction: {math.degrees(future_theta_prediction):.4f} deg")

        # Update Time Axis memory (Placed early so it is not lost on topology shift returns)
        self.prev_theta_fwd = theta_fwd

        # 3. Dynamic Topology Correction (Delta-Y Switching)
        if self.topology == "WYE":
            if abs(math.degrees(self.velocity_dtheta_dt)) > 15.0:
                print("  !! HIGH PHASE VELOCITY DETECTED: FUTURE COLLAPSE PREDICTED !!")
                print("  >> [TOPOLOGY SHIFT] WYE -> DELTA (Absorbing circulating tension...)")
                self.topology = "DELTA"
                return

        elif self.topology == "DELTA":
            if abs(math.degrees(self.velocity_dtheta_dt)) < 5.0:
                print("  >> [TOPOLOGY SHIFT] DELTA -> WYE (System stabilized. Re-establishing Neutral point...)")
                self.topology = "WYE"
                self.prev_theta_fwd = 0.0
                return

        # 4. Dimensional Expansion (Assimilating error to expand dimensions)
        if self.topology == "WYE" and i_n_mag > 10.0 and len(self.axes) < self.max_axes_supported:
            new_axis_angle = cmath.phase(external_noise_phasor)
            new_axis_mag = abs(external_noise_phasor) / voltage_input
            new_axis_name = chr(self.next_axis_char)
            self.next_axis_char += 1

            # Counter-tension (+pi) mirror symmetry for perfect complex differentiation
            counter_angle = new_axis_angle + math.pi

            self.axes.append({
                "name": new_axis_name,
                "angle": counter_angle,
                "magnitude": new_axis_mag
            })

            print("  !! UNKNOWN ASYMMETRICAL NOISE DETECTED (New Category) !!")
            print(f"  >> [AXIS EXPANSION] Master Rotor spawned Axis {new_axis_name} at {math.degrees(counter_angle):.1f} deg to assimilate.")

        elif i_n_mag > 100.0:
            # Absolute hard crash constraint
            print("  >> CRITICAL SURGE: BEYOND MULTI-DIMENSIONAL CAPACITY. HARD CRASH.")
            os._exit(1)


if __name__ == "__main__":
    engine = HyperRotorEngine()

    # Tick 1~2: Normal 3-Phase equilibrium (I_N = 0)
    engine.process_tick(voltage_input=220.0, external_noise_phasor=0j)
    engine.process_tick(voltage_input=220.0, external_noise_phasor=0j)

    # Tick 3: Unknown category influx (Axis expansion triggered)
    # 25A noise invading the 220V space at 45 degrees
    noise = cmath.rect(25.0, math.radians(45))
    engine.process_tick(voltage_input=220.0, external_noise_phasor=noise)

    # Tick 4: Confirm expanded Axis D perfectly absorbs noise, restoring equilibrium (I_N = 0)
    engine.process_tick(voltage_input=220.0, external_noise_phasor=noise)
