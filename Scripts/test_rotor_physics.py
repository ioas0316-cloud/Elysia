
import sys
import os
import math

# Add the project root to sys.path
sys.path.append(os.getcwd())

from Core.Keystone.sovereign_math import SovereignVector, DoubleHelixRotor

def test_rotor_acceleration():
    print("Testing DoubleHelixRotor Acceleration and Precession...")

    # Initialize a rotor in a specific plane (1, 2)
    rotor = DoubleHelixRotor(angle=0.1, p1=1, p2=2)

    # Initial state
    v = SovereignVector.ones().normalize()

    print(f"Initial Friction Vortex: {rotor.friction_vortex}")
    print(f"Initial Angular Momentum: {rotor.angular_momentum}")

    # Apply duality a few times
    for i in range(5):
        v = rotor.apply_duality(v)
        print(f"Step {i}: Friction: {rotor.friction_vortex:.4f}, Momentum: {rotor.angular_momentum:.4f}")

    # Test External Torque
    print("\nApplying External Torque (Noise)...")
    rotor.apply_external_torque(noise_intensity=2.0, tilt_delta=0.05)
    print(f"After Torque - Momentum: {rotor.angular_momentum:.4f}, Tilt: {rotor.precession_tilt:.4f}")

    v = rotor.apply_duality(v)
    print(f"After Torque Step: Friction: {rotor.friction_vortex:.4f}, Momentum: {rotor.angular_momentum:.4f}")

if __name__ == "__main__":
    test_rotor_acceleration()
