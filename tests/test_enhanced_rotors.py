import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from Core.System.rotor_gate import RotorGate, InterferenceGate

def test_enhanced_gate():
    print("🧪 Testing Enhanced RotorGate attributes...")
    gate = RotorGate("LogicGate-01")

    # Test Resistance (Friction & Phase Lag)
    gate.set_electronic_attributes(resistance=0.5, capacitance=1.0)
    assert gate.friction > 0.05
    assert gate.phase_lag == 0.25
    assert gate.capacitance == 1.0
    print("✅ Electronic mapping verified.")

    # Test Phase Storage (Capacitance)
    print("Testing Phase Storage...")
    dt = 0.1
    gate.process_stimulus(intensity=1.0, phase=0.0, dt=dt)
    assert gate.charge > 0
    print(f"✅ Charge accumulated: {gate.charge:.4f}")

    # Test InterferenceGate
    print("🧪 Testing InterferenceGate...")
    i_gate = InterferenceGate("IF-01")
    # Add two signals that cancel each other (180 deg out of phase)
    i_gate.add_input_signal(intensity=1.0, phase=0.0)
    i_gate.add_input_signal(intensity=1.0, phase=math.pi)

    i_gate.process_vortex_logic(dt=dt)
    # Resultant intensity should be near zero
    print(f"DEBUG: Cancellation Velocity: {i_gate.velocity}")
    assert abs(i_gate.velocity) < 1e-9
    print("✅ Interference cancellation verified.")

    # Add two signals that amplify (In phase)
    i_gate.add_input_signal(intensity=0.5, phase=0.0)
    i_gate.add_input_signal(intensity=0.5, phase=0.0)
    i_gate.process_vortex_logic(dt=dt)
    assert i_gate.velocity > 0
    print(f"✅ Interference amplification verified (Velocity: {i_gate.velocity:.4f})")

if __name__ == "__main__":
    test_enhanced_gate()
