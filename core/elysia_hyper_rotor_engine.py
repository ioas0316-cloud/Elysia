import cmath
import math

class HyperRotorEngine:
    def __init__(self, bypass_threshold=3):
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

        # System status flag and Bypass mechanics
        self.system_status = "NORMAL" # NORMAL, SURGE_CRITICAL, BYPASS
        self.bypass_threshold = bypass_threshold
        self.stable_ticks = 0
        self.bypass_mode = False

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

        # 1. Break Bypass Condition
        if self.bypass_mode and abs(external_noise_phasor) > self.epsilon:
            print(f"\n[Tick #{self.tick:04d}] =======================================================")
            print("!! ANOMALY DETECTED: BYPASS RELAY BROKEN. REVERTING TO VERIFICATION & LOGGING MODE !!")
            self.bypass_mode = False
            self.system_status = "NORMAL"
            self.stable_ticks = 0

        if self.bypass_mode:
            print(f"[Tick #{self.tick:04d}] [Master Rotor] BYPASS MODE ACTIVE: High-speed zero-overhead passing.")
            return

        print(f"\n[Tick #{self.tick:04d}] =======================================================")
        print(f"[Master Rotor] Evaluating {len(self.axes)}-Dimensional Hyper-Space. Topology: {self.topology} | Status: {self.system_status}")

        # 2. Forward Helix Tensor Evaluation
        tensor_fwd = self.calculate_system_tensor(voltage_input, external_noise_phasor)
        theta_fwd = cmath.phase(tensor_fwd) if abs(tensor_fwd) > self.epsilon else 0.0

        # 3. Phase Unwrapping (Mathematical Discontinuity Correction)
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

        # 4. Dynamic Topology Correction (Delta-Y Switching)
        if self.topology == "WYE":
            if abs(math.degrees(self.velocity_dtheta_dt)) > 15.0:
                print("  !! HIGH PHASE VELOCITY DETECTED: FUTURE COLLAPSE PREDICTED !!")
                print("  >> [TOPOLOGY SHIFT] WYE -> DELTA (Absorbing circulating tension...)")
                self.topology = "DELTA"
                self.stable_ticks = 0
                return

        elif self.topology == "DELTA":
            if abs(math.degrees(self.velocity_dtheta_dt)) < 5.0:
                print("  >> [TOPOLOGY SHIFT] DELTA -> WYE (System stabilized. Re-establishing Neutral point...)")
                self.topology = "WYE"
                self.prev_theta_fwd = 0.0
                self.stable_ticks = 0
                return

        # 5. Dimensional Expansion (Assimilating error to expand dimensions) and Limits
        if i_n_mag > 100.0:
            # System logs the surge and spins, no os._exit()
            self.system_status = "SURGE_CRITICAL"
            print("  !! WARNING !! >> CRITICAL SURGE: BEYOND MULTI-DIMENSIONAL CAPACITY.")
            print("  >> [STATUS] SYSTEM UNBALANCED BUT CONTINUING SPIN.")
            self.stable_ticks = 0

        elif self.topology == "WYE" and i_n_mag > 10.0 and len(self.axes) < self.max_axes_supported:
            self.system_status = "NORMAL"
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
            self.stable_ticks = 0

        else:
            self.system_status = "NORMAL"

        # 6. Check for Bypass stability
        if i_n_mag < self.epsilon and abs(math.degrees(self.velocity_dtheta_dt)) < 1.0 and self.topology == "WYE":
            self.stable_ticks += 1
            if self.stable_ticks >= self.bypass_threshold and not self.bypass_mode:
                self.bypass_mode = True
                self.system_status = "BYPASS"
                print("  >> [SYSTEM STATUS] INTEGRITY CONFIRMED. ALL TENSIONS BALANCED.")
                print("  >> [ACTION] ENABLING HIGH-SPEED BYPASS RELAY (ZERO-OVERHEAD MODE)\n")
        else:
            self.stable_ticks = 0


if __name__ == "__main__":
    pass
