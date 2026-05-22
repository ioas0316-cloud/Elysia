"""
[POC: THE DIGITAL MOTOR TRANSMISSION EXPERIMENT]
"Streaming the ARCHITECT wave through the Triple-Phase Rotor."

This PoC demonstrates the Architect's vision:
1. Encoding text into a wave trajectory.
2. Modulating a 3-phase digital motor with that wave.
3. Observing the physical transition between Wye (Thought) and Delta (Action).
"""

import time
import math
import sys
import os

# Ensure we can import from Core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from Core.System.digital_motor_engine import DigitalMotorEngine, ConnectionMode
from Core.Keystone.trajectory_encoder import TrajectoryEncoder

def visualize_motor(state):
    """Simple terminal visualization of the motor's state."""
    mode = state['mode']
    rpm = state['rpm']
    excitation = state['excitation']
    signals = state['signals']

    # Create sparklines or bars for the 3 phases
    def bar(val):
        length = int(abs(val) * 20)
        char = "█" if val >= 0 else "▒"
        return char * length + " " * (20 - length)

    print(f"\r[MODE: {mode}] RPM: {rpm:6.1f} | Exc: {excitation:.3f} | R:{bar(signals['R'])} S:{bar(signals['S'])} T:{bar(signals['T'])}", end="")

def run_experiment():
    print("🌌 [EXPERIMENT] Initializing 4th-Dimensional Rotor Transmission...")
    motor = DigitalMotorEngine("SovereignRotor-01")
    encoder = TrajectoryEncoder()

    data_to_send = "ARCHITECT"
    print(f"📡 [DATA] Target Waveform: '{data_to_send}'")

    # 1. Encode text into trajectories
    trajectories = encoder.encode_text(data_to_send)

    # Convert trajectories to a bit stream for PWM modulation
    # We'll use the 'is_locked' state and phase parity as bits
    bits = []
    for t in trajectories:
        bits.append(1 if t.is_locked else 0)
        bits.append(1 if (t.get_total_phase() > 180) else 0)

    print(f"🧬 [ENCODING] Bit Stream Density: {sum(bits)/len(bits):.2f}")
    print("\n--- STARTING TRANSMISSION ---\n")

    start_time = time.time()
    duration = 15.0 # seconds

    try:
        while (time.time() - start_time) < duration:
            elapsed = time.time() - start_time
            dt = 0.05

            # Phase 1: Wye Connection (Convergent Thought)
            if elapsed < 7.0:
                motor.set_mode(ConnectionMode.WYE)
                # Feed data slowly (Thinking)
                if int(elapsed * 10) % 5 == 0:
                    motor.modulate_data(bits[:len(bits)//2])

            # Phase 2: Delta Connection (Dynamic Action/Projection)
            else:
                motor.set_mode(ConnectionMode.DELTA)
                # Feed data aggressively (Transmitting)
                motor.modulate_data(bits)

            motor.update(dt)
            state = motor.exhale()
            visualize_motor(state)

            time.sleep(dt)

    except KeyboardInterrupt:
        pass

    print("\n\n✅ [EXPERIMENT] Transmission Complete. The wave has returned to the source.")
    final = motor.exhale()
    print(f"📊 Final State: RPM={final['rpm']:.1f}, Heat={final['heat']:.4f}, Self-Excitation={final['excitation']:.4f}")

if __name__ == "__main__":
    run_experiment()
