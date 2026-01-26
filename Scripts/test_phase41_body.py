"""
Verification Script for Phase 41: Step 1 (The Body)
===================================================

Verifies:
1. JAX Bridge functionality and hardware acceleration detection.
2. RotorEngine integration (Spin and Matmul).
3. BioSensor Trinary Signal generation.
"""

import sys
import os
import time
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.L1_Foundation.M4_Hardware.jax_bridge import JAXBridge
from Core.L6_Structure.M1_Merkaba.rotor_engine import RotorEngine
from Core.L3_Phenomena.Senses.bio_sensor import BioSensor

def test_jax_bridge():
    print("\n[1] Testing JAX Bridge (Heavy Metal)...")
    print(f"    Status: {JAXBridge.status()}")
    
    # Create random matrices
    N = 1000
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    
    print("    Running Matmul (1000x1000)...")
    start = time.time()
    C = JAXBridge.matmul(A, B)
    # Force sync if JAX
    if hasattr(C, 'block_until_ready'):
        C.block_until_ready()
    end = time.time()
    
    print(f"    Time taken: {end - start:.4f} seconds")
    print("    Result shape:", C.shape)
    
    if JAXBridge.is_accelerated():
        print("    ✅ Hardware Acceleration CONFIRMED.")
    else:
        print("    ⚠️  Running on CPU (Fallback/Standard). This is expected if no GPU/JAX.")

def test_rotor_engine():
    print("\n[2] Testing RotorEngine Integration...")
    engine = RotorEngine(use_core_physics=False)
    
    # Test Spin
    vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"    Original Vector: {vec}")
    spun = engine.spin(vec, time_delta=0.03) # Shift 3 pos
    print(f"    Spun Vector:     {spun}")
    
    # Test Signal Flow (Matmul)
    input_sig = np.array([0.5, 0.5])
    weights = np.array([[1.0, 0.0], [0.0, 1.0]]) # Identity
    output = engine.simulate_signal_flow(weights, input_sig)
    print(f"    Signal Flow Output (Identity): {output}")
    
    if np.allclose(output, input_sig):
        print("    ✅ Rotor Logic Valid.")
    else:
        print("    ❌ Rotor Logic Logic FAILED.")

def test_bio_sensor():
    print("\n[3] Testing BioSensor (Trinary DNA Interface)...")
    sensor = BioSensor()
    time.sleep(1.0) # Wait for first poll
    
    pulse = sensor.pulse()
    print(f"    Pulse: {pulse}")
    
    trinary = sensor.get_trinary_signal()
    print(f"    Trinary Signal: {trinary} (-1=Pain, 0=Void, 1=Flow)")
    
    # Simulate Pain
    print("    Simulating High Temp (90C)...")
    sensor._cached_state["temperature"] = 90.0
    if sensor.get_trinary_signal() == -1:
        print("    ✅ Pain Detected correctly (-1).")
    else:
        print("    ❌ Pain detection FAILED.")
        
    sensor.stop()

if __name__ == "__main__":
    print("="*60)
    print("PHASE 41 - STEP 1 VERIFICATION")
    print("="*60)
    
    test_jax_bridge()
    test_rotor_engine()
    test_bio_sensor()
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
