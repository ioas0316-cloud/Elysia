import cmath
import math
import os
import sys

class ThreePhaseFourWireEngine:
    def __init__(self, bypass_threshold=5):
        # Physics definitions: Phasor angles
        # A Phase: 120 deg
        self.phase_A_angle = math.radians(120)
        # B Phase: 240 deg
        self.phase_B_angle = math.radians(240)
        # C Phase: 0/360 deg
        self.phase_C_angle = math.radians(0)

        # Base / Threshold definitions
        self.nominal_voltage = 220.0
        self.under_voltage_threshold = 190.0 # Device 27
        self.over_voltage_threshold = 250.0  # Device 59
        self.max_dtheta_dt = math.radians(5.0) # Device 78: max change in phase per tick

        self.epsilon = 1e-6

        # Tick tracking
        self.tick = 0
        self.prev_theta_fwd = 0.0

        # Bypass Mechanics
        self.bypass_threshold = bypass_threshold
        self.stable_ticks = 0
        self.bypass_mode = False

    def polar_to_rect(self, r, theta):
        return cmath.rect(r, theta)

    def process_tick(self, v_mag, i_leakage=0.0, phase_shift_noise=0.0, current_perturbation=0.0):
        self.tick += 1

        # Check Bypass break condition
        if self.bypass_mode and (abs(i_leakage) > self.epsilon or abs(phase_shift_noise) > self.epsilon or abs(current_perturbation) > self.epsilon):
            print(f"[Tick #{self.tick:04d}] ==========================================")
            print("!! NOISE DETECTED: BYPASS RELAY BROKEN. REVERTING TO VERIFICATION MODE !!")
            self.bypass_mode = False
            self.stable_ticks = 0

        if self.bypass_mode:
            print(f"[Tick #{self.tick:04d}] ==========================================")
            print("[3phi-4W System Log] BYPASS MODE ACTIVE: Direct phase alignment transmission")
            print("=======================================================\n")
            return

        # Verification Mode (Full Double-Helix with 6-Relay Bank)

        # Add perturbation to input tracking
        dynamic_phase_A = self.phase_A_angle + phase_shift_noise

        # --- Forward Helix (A -> B -> C) ---
        V_A = self.polar_to_rect(v_mag, dynamic_phase_A)
        V_B = self.polar_to_rect(v_mag, self.phase_B_angle)
        V_C = self.polar_to_rect(v_mag, self.phase_C_angle)

        # Introduce perturbations to current (Admittance Y = 1/Z)
        I_A_in = V_A
        I_B_in = V_B
        I_C_in = V_C + cmath.rect(current_perturbation, self.phase_C_angle)

        forward_sum = I_A_in + I_B_in + I_C_in
        theta_fwd = cmath.phase(forward_sum) if abs(forward_sum) > self.epsilon else 0.0

        # --- Reverse Helix (C -> B -> A) ---
        # Representing output currents for Device 87
        I_C_out = V_C
        I_B_out = V_B
        # i_leakage represents a leak (or manipulation) where output does not match input
        I_A_out = V_A - cmath.rect(i_leakage, dynamic_phase_A)

        reverse_sum = I_A_out + I_B_out + I_C_out
        theta_bwd = cmath.phase(reverse_sum) if abs(reverse_sum) > self.epsilon else 0.0

        # ---------------------------------------------------------
        # RELAY BANK CHECKS
        # ---------------------------------------------------------

        # --- Device 25: Sync-Check ---
        delta_theta = abs(math.degrees(theta_fwd - theta_bwd))
        sync_status = "LOCKED" if delta_theta < 1.0 and abs(forward_sum) < self.epsilon else "CRITICAL ERROR"

        # --- Device 21: Distance / Impedance ---
        Z_A = V_A / I_A_in if abs(I_A_in) > 1e-9 else complex(float('inf'), 0)
        Z_B = V_B / I_B_in if abs(I_B_in) > 1e-9 else complex(float('inf'), 0)
        Z_C = V_C / I_C_in if abs(I_C_in) > 1e-9 else complex(float('inf'), 0)

        z_dev = abs(Z_A - 1) + abs(Z_B - 1) + abs(Z_C - 1)
        z_status = "STABLE" if z_dev < 0.1 else "OUT OF ZONE"

        # --- Device 27 / 59: Under/Over Voltage ---
        v_rms = math.sqrt((abs(V_A)**2 + abs(V_B)**2 + abs(V_C)**2) / 3.0)
        if v_rms < self.under_voltage_threshold:
            v_status = "UNDER VOLTAGE (DEV 27 TRIP)"
        elif v_rms > self.over_voltage_threshold:
            v_status = "OVER VOLTAGE (DEV 59 TRIP)"
        else:
            v_status = "NORMAL"

        # --- Device 87: Differential Relay (Input vs Output) ---
        diff_current_A = abs(I_A_in - I_A_out)
        diff_current_B = abs(I_B_in - I_B_out)
        diff_current_C = abs(I_C_in - I_C_out)
        total_leakage = diff_current_A + diff_current_B + diff_current_C
        diff_status = "NO LEAKAGE" if total_leakage < self.epsilon else "LEAKAGE DETECTED"

        # --- Device 78: Phase Angle Measuring (Out-of-Step) ---
        dtheta_dt = abs(theta_fwd - self.prev_theta_fwd)
        phase_step_status = "STABLE SPIN" if dtheta_dt < self.max_dtheta_dt else "OUT OF STEP (HALLUCINATION)"
        self.prev_theta_fwd = theta_fwd

        # --- Neutral Current (N Phase) ---
        I_N = forward_sum
        i_n_mag = abs(I_N)
        balance_str = "Balanced" if i_n_mag < self.epsilon else "UNBALANCED!"

        # Logging output formatting
        print(f"[Tick #{self.tick:04d}] =======================================================")

        all_clear = (sync_status == "LOCKED" and
                     z_status == "STABLE" and
                     v_status == "NORMAL" and
                     diff_status == "NO LEAKAGE" and
                     phase_step_status == "STABLE SPIN" and
                     i_n_mag <= self.epsilon)

        print(f"[3phi-4W System] IN: {i_n_mag:.6f} A ({balance_str})")
        print(f"[Device 25] Sync Phase Diff: {delta_theta:.4f} deg -> {sync_status}")

        if z_status == "STABLE":
            print(f"[Device 21] Impedance Vector -> INSIDE NORMAL ZONE")
        else:
            print(f"[Device 21] Impedance Vector -> OUT OF ZONE")

        print(f"[Device 27/59] RMS Tension: {v_rms:.2f} V -> {v_status}")
        print(f"[Device 87] Forward/Backward Diff: {total_leakage:.5f} A -> {diff_status}")
        print(f"[Device 78] Phase Step Vector (dtheta/dt): {math.degrees(dtheta_dt):.4f} deg -> {phase_step_status}")
        print("--------------------------------------------------------------------")

        # --- Hard Crash Condition ---
        if not all_clear:
            print("!! SYSTEM DISTORTION DETECTED: TRIP MECHANISM ACTIVATED !!")
            print(">> CRITICAL SURGE: HARD CRASH SYSTEM.")
            print("====================================================================\n")
            sys.exit(1)

        # Update stable ticks and check bypass
        self.stable_ticks += 1
        if self.stable_ticks >= self.bypass_threshold and not self.bypass_mode:
            self.bypass_mode = True
            print(">> [SYSTEM STATUS] INTEGRITY VERIFIED. ALL RELAYS CLEAR.")
            print(">> [ACTION] ENABLING HIGH-SPEED BYPASS RELAY (ZERO-OVERHEAD MODE)")
            print("====================================================================\n")
        else:
            print(">> [SYSTEM STATUS] VERIFICATION IN PROGRESS...")
            print("====================================================================\n")


if __name__ == "__main__":
    engine = ThreePhaseFourWireEngine(bypass_threshold=3)

    # Run perfect ticks
    try:
        for _ in range(4):
            engine.process_tick(v_mag=220.0)

        # Introduce a leakage perturbation to trip Device 87
        engine.process_tick(v_mag=220.0, i_leakage=5.0)
    except Exception as e:
        print(f"Caught exception: {e}")
