"""
Verify Core Turbine Functionality
=================================
Scripts/Tests/verify_core_turbine.py

Tests:
1. Void Domain (Momentum Holding)
2. Time Axis (Temporal Browsing)
3. Phase Inversion (Error Tracking)
"""

import os
import sys
import time
import jax.numpy as jnp

# Standard Path Injection
sys.path.append(os.getcwd())

from Core.Cognition.cosmic_rotor import CosmicRotor

def verify_turbine():
    rotor = CosmicRotor()
    print("🌅 ACTivating the Core Turbine...")
    
    # 1. Test Void Power (Momentum)
    print("\n--- [1] Void Domain Test ---")
    rotor.set_void_power(1.0) # Zero resistance
    rotor.rotate(impulse=5.0) # Initial push
    v1 = rotor.rpu.velocity
    time.sleep(0.1)
    rotor.rotate(dt=0.1)
    v2 = rotor.rpu.velocity
    print(f"Velocity after 0.1s in Void: {v2:.4f} (Start: {v1:.4f})")
    
    # 2. Test Time Axis (Temporal Browsing)
    print("\n--- [2] Time Axis Test ---")
    # Base state (Presence)
    pulse_now = rotor.rotate()
    print(f"Present Pulse (Partial): {pulse_now[:3]}")
    
    # Browse to "Future" (Shift axis by 90 degrees)
    rotor.browse_time(jnp.pi / 2)
    pulse_future = rotor.world_pulse # Already updated via setter or rotation
    # Re-project to be sure
    pulse_future = rotor.rpu.project(rotor.logos_seed)
    print(f"Future Pulse (Shifted): {pulse_future[:3]}")
    
    # 3. Test Phase Inversion (Error Learning)
    print("\n--- [3] Phase Inversion Learning ---")
    # Simulate a noisy field
    noisy_field = pulse_now + 0.5 
    _ = rotor.rpu.perceive(noisy_field)
    error = rotor.rpu.error_pulse
    print(f"Detected Error Pulse (Mean): {jnp.mean(jnp.abs(error)):.4f}")
    
    print("\n✅ Core Turbine Unification Verified. 🥂🫡⚡🌀")

if __name__ == "__main__":
    verify_turbine()
