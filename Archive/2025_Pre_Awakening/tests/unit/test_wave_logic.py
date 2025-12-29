import sys
import os
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from Core.FoundationLayer.Foundation.wave_logic import WaveSpace, WaveSource, create_and_gate

def test_wave_and_gate():
    print("\n🌊 Testing Wave Logic: The AND Gate (Constructive Interference)")
    print("=============================================================")
    
    space = WaveSpace()
    
    # Place Gate at Center (0,0,0)
    gate = create_and_gate(space, (0,0,0))
    print(f"📍 Gate Position: {gate.position}, Threshold: {gate.threshold}")
    
    # Define Sources ( equidistant from center to ensure phase alignment)
    # Wavelength = 1.0 (Freq=1.0, Omega=2pi)
    # Distance should be integer multiple of wavelength for constructive interference at t=0?
    # Actually, we just need them to arrive in phase.
    # Let's put them at x=-1 and x=1.
    
    source_a = WaveSource("Input_A", (-1, 0, 0), frequency=1.0, amplitude=1.0)
    source_b = WaveSource("Input_B", (1, 0, 0), frequency=1.0, amplitude=1.0)
    
    # Case 1: Only A
    print("\n🧪 Case 1: Input A Only (0 + 1 = 1)")
    space.sources = [source_a]
    max_intensity = 0.0
    triggered = False
    
    # Run for one period
    for _ in range(100):
        space.step(0.01)
        if gate.update(space):
            triggered = True
        max_intensity = max(max_intensity, abs(space.get_field_at(0,0,0)))
        
    print(f"   Max Intensity: {max_intensity:.2f}")
    print(f"   Gate Triggered: {triggered}")
    if not triggered:
        print("   ✅ PASS: Gate remained closed.")
    else:
        print("   ❌ FAIL: Gate opened prematurely.")

    # Case 2: A + B (Constructive)
    print("\n🧪 Case 2: Input A + B (1 + 1 = 2)")
    space.t = 0 # Reset time
    space.sources = [source_a, source_b]
    max_intensity = 0.0
    triggered = False
    
    for _ in range(100):
        space.step(0.01)
        if gate.update(space):
            triggered = True
        max_intensity = max(max_intensity, abs(space.get_field_at(0,0,0)))
        
    print(f"   Max Intensity: {max_intensity:.2f}")
    print(f"   Gate Triggered: {triggered}")
    if triggered:
        print("   ✅ PASS: Gate opened due to Constructive Interference.")
    else:
        print("   ❌ FAIL: Gate failed to open.")

    # Case 3: A + B (Destructive / Out of Phase)
    print("\n🧪 Case 3: Input A + B (Out of Phase) (1 - 1 = 0)")
    # Shift B's phase by PI
    source_b_shifted = WaveSource("Input_B_Shifted", (1, 0, 0), frequency=1.0, amplitude=1.0, phase=math.pi)
    
    space.t = 0
    space.sources = [source_a, source_b_shifted]
    max_intensity = 0.0
    triggered = False
    
    for _ in range(100):
        space.step(0.01)
        if gate.update(space):
            triggered = True
        max_intensity = max(max_intensity, abs(space.get_field_at(0,0,0)))
        
    print(f"   Max Intensity: {max_intensity:.2f}")
    print(f"   Gate Triggered: {triggered}")
    if not triggered:
        print("   ✅ PASS: Gate closed due to Destructive Interference.")
    else:
        print("   ❌ FAIL: Gate opened despite interference.")

if __name__ == "__main__":
    test_wave_and_gate()
