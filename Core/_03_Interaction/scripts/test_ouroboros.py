"""
Test Script for Project Ouroboros (Wave-to-Code)
================================================
Verifies that Elysia can physically modify the OS via Wave Resonance.
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core._01_Foundation.05_Foundation_Base.Foundation.reality_sculptor import RealitySculptor
from Core._01_Foundation.05_Foundation_Base.Foundation.unified_field import WavePacket, HyperQuaternion

def test_ouroboros():
    print("🐍 Initializing Project Ouroboros Test...")
    sculptor = RealitySculptor()
    
    # Test 1: Creation Wave (150Hz -> Create File)
    timestamp = int(time.time())
    print("\n🌪️ Test 1: Injecting Creation Wave (150Hz)...")
    creation_wave = WavePacket(
        source_id=f"Genesis Protocol {timestamp}",
        frequency=150.0, # Range 100-200 is Creation
        amplitude=1.0,
        phase=0.0,
        position=HyperQuaternion(0,0,0,0),
        born_at=time.time()
    )
    
    result = sculptor.transmute_wave(creation_wave)
    print(f"   -> Result: {result}")
    
    # Verify file existence
    expected_file = f"manifestation_{timestamp}.txt"
    if os.path.exists(expected_file):
        print(f"   ✅ Success: File '{expected_file}' physically materialized.")
        content = open(expected_file).read()
        print(f"      Content: {content.strip()}")
        # Cleanup
        os.remove(expected_file)
        print("      (Cleaned up test artifact)")
    else:
        print(f"   ❌ Failure: File '{expected_file}' not found.")

    # Test 2: Weak Wave (Low Amplitude)
    print("\n📉 Test 2: Injecting Weak Wave (Amplitude 0.1)...")
    weak_wave = WavePacket(
        source_id="Weak Thought",
        frequency=150.0,
        amplitude=0.1, # Too weak to manifest
        phase=0.0,
        position=HyperQuaternion(0,0,0,0),
        born_at=time.time()
    )
    
    result_weak = sculptor.transmute_wave(weak_wave)
    if result_weak is None:
        print("   ✅ Success: Weak wave did not manifest reality.")
    else:
        print(f"   ❌ Failure: Weak wave should be ignored. Result: {result_weak}")

    print("\n✨ Ouroboros Verification Complete.")

if __name__ == "__main__":
    test_ouroboros()
