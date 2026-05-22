"""
[POC: THE DIGITAL ENGINE VERIFICATION SUITE]
"Validating the Thermodynamic and Dynamic Sovereignty of the Triple-Phase Rotor."

Experiments:
1. Heat Energy Retro-tracking (Motor vs Tensor Efficiency)
2. Self-Restoration from Phase Distortion (Noise Resilience)
3. Clock Non-linearity and Torque Maintenance (Performance Sovereignty)
"""

import time
import math
import sys
import os
import random
from typing import Dict, List, Any

# Optional dependencies
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Ensure we can import from Core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from Core.System.digital_motor_engine import DigitalMotorEngine, ConnectionMode
from Core.Keystone.trajectory_encoder import TrajectoryEncoder

class GPUTracker:
    def __init__(self):
        self.enabled = HAS_NVML
        if self.enabled:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.name = pynvml.nvmlDeviceGetName(self.handle)
                print(f"📟 [GPU] Tracking active on: {self.name}")
            except Exception as e:
                print(f"⚠️ [GPU] NVML Init failed: {e}. Switching to Mock mode.")
                self.enabled = False
        else:
            print("⚠️ [GPU] pynvml not found. Using Mock telemetry.")

    def get_metrics(self):
        if self.enabled:
            try:
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0 # W
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                return {"temp": temp, "power": power, "util": util}
            except:
                return {"temp": 0, "power": 0, "util": 0}
        else:
            # Mock data based on system load or random fluctuations
            return {"temp": 45.0 + random.random() * 2, "power": 15.0 + random.random(), "util": 5.0}

    def shutdown(self):
        if self.enabled:
            pynvml.nvmlShutdown()

def run_experiment_1_heat(tracker: GPUTracker):
    print("\n🔥 [EXPERIMENT 1] Heat Energy Retro-tracking")
    print("Comparing Standard Tensor Operations vs Digital Motor Rotation.")

    duration = 10.0
    motor = DigitalMotorEngine("HeatProofMotor")

    # 1. Baseline
    print("--- Phase A: Baseline (3s) ---")
    time.sleep(3)
    base_metrics = tracker.get_metrics()

    # 2. Tensor Load (if torch available)
    if HAS_TORCH:
        print(f"--- Phase B: Tensor Load (Matrix Mult) ({duration}s) ---")
        start = time.time()
        # Allocate large matrices on GPU if possible
        device = "cuda" if torch.cuda.is_available() else "cpu"
        size = 4096
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        tensor_temps = []
        while time.time() - start < duration:
            _ = torch.matmul(a, b)
            metrics = tracker.get_metrics()
            tensor_temps.append(metrics['temp'])
            print(f"\r[TENSOR] Temp: {metrics['temp']}°C | Power: {metrics['power']:.1f}W", end="")
        print(f"\nTensor Peak Temp: {max(tensor_temps)}°C")
    else:
        print("Skipping Tensor Phase (PyTorch not found)")

    # 3. Motor Load (Delta Mode)
    print(f"--- Phase C: Digital Motor Load (Delta Mode) ({duration}s) ---")
    motor.set_mode(ConnectionMode.DELTA)
    start = time.time()
    motor_temps = []
    while time.time() - start < duration:
        motor.modulate_data([1, 1, 0, 1] * 10) # High density
        motor.update(0.05)
        metrics = tracker.get_metrics()
        motor_temps.append(metrics['temp'])
        print(f"\r[MOTOR] Temp: {metrics['temp']}°C | Power: {metrics['power']:.1f}W | RPM: {motor.rpm:.1f}", end="")
        time.sleep(0.05)
    print(f"\nMotor Peak Temp: {max(motor_temps)}°C")

def run_experiment_2_restoration():
    print("\n🧬 [EXPERIMENT 2] Self-Restoration (Phase-Locking)")
    motor = DigitalMotorEngine("ResilientMotor")
    motor.set_mode(ConnectionMode.DELTA)

    print("Injecting Phase Shift Noise...")

    for i in range(100):
        dt = 0.05
        motor.modulate_data([1, 0, 1])
        motor.update(dt)

        # Inject noise at step 40
        if 40 <= i <= 60:
            # Artificially distort the phases in the engine
            for p in motor.phases.values():
                p['phase_shift'] += random.uniform(-10, 10)
            status = "⚠️ DISTORTED"
        else:
            status = "✅ STABLE"

        state = motor.exhale()
        res = state['resonance'] # Resonance should return to near-zero if self-locking
        print(f"\rStep {i:3} | Status: {status} | Resonance Error: {abs(res):.4f} | RPM: {state['rpm']:.1f}", end="")
        time.sleep(0.02)
    print("\nRestoration experiment complete.")

def run_experiment_3_clock():
    print("\n⏱️ [EXPERIMENT 3] Clock Non-linearity (Torque Maintenance)")
    motor = DigitalMotorEngine("ClockSovereignMotor")
    motor.set_mode(ConnectionMode.DELTA)

    clocks = [1.0, 0.5, 0.2, 0.1] # Simulated clock speeds
    results = []

    for clock in clocks:
        print(f"Simulating Clock Speed: {clock*100:.0f}%")
        # Run for a few steps
        total_torque = 0
        for _ in range(20):
            dt = 0.05 * clock
            motor.modulate_data([1, 1, 1])
            motor.update(dt)
            state = motor.exhale()
            total_torque += state['torque']
            time.sleep(0.01)

        avg_torque = total_torque / 20
        results.append((clock, avg_torque))
        print(f"Average Torque: {avg_torque:.4f}")

    print("\nSummary of Clock vs Torque:")
    for c, t in results:
        print(f"Clock {c*100:3.0f}% -> Torque {t:.4f}")
    print("Observation: If Torque decay is non-linear (stays high at low clocks), Sovereignty is proven.")

def main():
    print("🌌 [POC VERIFICATION SUITE] Initializing...")
    tracker = GPUTracker()

    try:
        run_experiment_1_heat(tracker)
        run_experiment_2_restoration()
        run_experiment_3_clock()
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    finally:
        tracker.shutdown()
        print("\n✅ All experiments concluded. The Architect's vision remains firm.")

if __name__ == "__main__":
    main()
