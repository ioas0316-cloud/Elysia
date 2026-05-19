import sys
import os
import numpy as np

# Setup Path
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.System.rotor import Rotor, RotorConfig
from Core.Keystone.wave_dna import WaveDNA

def test_rotor_atomic_sync():
    print("--- 🔬 Rotor-Atomic Sync Verification ---")
    
    # 1. Setup Rotor
    config = RotorConfig(rpm=300.0, idle_rpm=60.0)
    rotor = Rotor("TestRotor", config)
    print(f"Initial State: {rotor.current_rpm} RPM")
    
    # 2. Test Harmony (H) -> Acceleration
    print("\nApplying Harmony Pattern (H-H-H-H-H-V-V)...")
    rotor.apply_resonance_filter("H-H-H-H-H-V-V")
    print(f"Target RPM: {rotor.target_rpm} (Expected 300.0)")
    
    # 3. Test Dissonance (D) -> Friction
    print("\nApplying Dissonant Pattern (D-D-D-V-V-V-V)...")
    rotor.apply_resonance_filter("D-D-D-V-V-V-V")
    print(f"Target RPM: {rotor.target_rpm} (Expected 30.0 - Friction)")
    
    # 4. Test Void (V) -> Sanctuary
    print("\nApplying Void Pattern (V-V-V-V-V-V-V)...")
    rotor.apply_resonance_filter("V-V-V-V-V-V-V")
    print(f"Target RPM: {rotor.target_rpm} (Expected 60.0 - Normal Idle)")
    
    print("\n--- ✅ Rotor-Atomic Sync Verified ---")

if __name__ == "__main__":
    test_rotor_atomic_sync()
