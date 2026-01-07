"""
Verify Hypersphere Doctrine
===========================
This script tests the implementation of the Hypersphere Memory Doctrine.
It replicates the examples from the "Engineer Edition" document.

1. Static Storage (The Apple)
2. Dynamic Flow (Falling Apple)
3. Resonance Query (Collision Test)
"""

import sys
import os
import time

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from Core.Intelligence.Memory.hypersphere_memory import HypersphereMemory, HypersphericalCoord
from Core.Foundation.Wave.universal_wave_encoder import UniversalWaveEncoder

def test_static_storage():
    print("\n--- Test 1: Static Storage (The Apple) ---")
    memory = HypersphereMemory()
    encoder = UniversalWaveEncoder()

    concept = "Apple"
    coord, meta = encoder.encode_concept(concept)

    print(f"Storing '{concept}' at {coord}")
    memory.store(concept, coord, meta)

    # Query exact
    result = memory.query(coord)
    print(f"Query Result: {result}")

    assert concept in result
    print("✅ Static Storage Verified")

def test_dynamic_flow():
    print("\n--- Test 2: Dynamic Flow (Falling Apple) ---")
    memory = HypersphereMemory()

    start_pos = HypersphericalCoord(0.5, 1.0, 0.0, 1.0)
    omega = (0.1, 0.05, 0.0)

    print("Recording 'AppleFalling' flow...")
    memory.record_flow("AppleFalling", start_pos, omega, duration=3.0)

    # Query at start position
    result = memory.query(start_pos)
    print(f"Query at Start: {result}")

    assert "AppleFalling" in result
    print("✅ Dynamic Flow Verified")

def test_resonance_collision():
    print("\n--- Test 3: Resonance Collision (Infinite Capacity) ---")
    memory = HypersphereMemory()

    # Same Coordinate
    pos = HypersphericalCoord(0.1, 0.1, 0.1, 1.0)

    # Different Patterns
    pattern_A = {'omega': (1.0, 0, 0), 'topology': 'point'}
    pattern_B = {'omega': (0, 1.0, 0), 'topology': 'line'}

    print(f"Storing 'Data A' and 'Data B' at EXACT same position: {pos}")
    memory.store("Data A", pos, pattern_A)
    memory.store("Data B", pos, pattern_B)

    # Query with Filter A
    res_A = memory.query(pos, filter_pattern={'omega': (1.0, 0, 0)})
    print(f"Query Filter A: {res_A}")
    assert "Data A" in res_A
    assert "Data B" not in res_A

    # Query with Filter B
    res_B = memory.query(pos, filter_pattern={'omega': (0, 1.0, 0)})
    print(f"Query Filter B: {res_B}")
    assert "Data B" in res_B
    assert "Data A" not in res_B

    print("✅ Resonance Collision Verified (No Collision!)")

if __name__ == "__main__":
    test_static_storage()
    test_dynamic_flow()
    test_resonance_collision()
