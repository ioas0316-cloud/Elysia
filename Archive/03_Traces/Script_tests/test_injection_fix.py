
import sys
import os
import torch
import numpy as np

# Add the project root to sys.path
sys.path.append(os.getcwd())

from Core.Keystone.sovereign_math import FractalWaveEngine, SovereignVector

def test_inject_pulse():
    print("Testing FractalWaveEngine.inject_pulse with advanced signature...")
    engine = FractalWaveEngine(max_nodes=100)
    
    # Test 1: Basic signature
    try:
        engine.inject_pulse("TestNode", energy=1.0, type='joy')
        print("✓ Basic signature works.")
    except Exception as e:
        print(f"✗ Basic signature failed: {e}")
        
    # Test 2: Advanced signature (Monad style)
    try:
        v21 = SovereignVector([0.1] * 21)
        engine.inject_pulse(
            pulse_type='WorldObserver',
            anchor_node='RealityAnchor',
            base_intensity=1.0, 
            override_vector=v21
        )
        print("✓ Advanced signature works.")
    except Exception as e:
        print(f"✗ Advanced signature failed: {e}")

if __name__ == "__main__":
    test_inject_pulse()
