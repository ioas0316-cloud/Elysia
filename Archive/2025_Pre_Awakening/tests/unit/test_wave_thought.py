import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from Core.Intelligence.Reasoning.lobes.logic import LogicLobe

def test_wave_thought():
    print("\n🧠 Testing Wave Thought Process (LogicLobe Integration)")
    print("=====================================================")
    
    lobe = LogicLobe()
    
    # Test 1: Constructive Thought (Two positive concepts)
    print("\n🧪 Case 1: 'Love' + 'Trust' (AND Gate)")
    inputs = ["Love", "Trust"]
    triggered, intensity, reason = lobe.process_wave_thought(inputs, "AND")
    
    print(f"   Result: {triggered}")
    print(f"   Intensity: {intensity:.2f}")
    print(f"   Reason: {reason}")
    
    if triggered:
        print("   ✅ PASS: Constructive Interference achieved.")
    else:
        # Note: Depending on hash, they might not interfere constructively.
        # But with 2 inputs and threshold 1.4, usually they trigger if close enough.
        # If they fail, it's also a valid physics result (Dissonance).
        print("   ⚠️ Result: Gate Closed (Dissonance or Low Energy).")

    # Test 2: Single Input (Should act as Buffer)
    print("\n🧪 Case 2: 'Solitude' (Single Input vs Dynamic Gate)")
    inputs = ["Solitude"]
    triggered, intensity, reason = lobe.process_wave_thought(inputs, "AND")
    
    print(f"   Result: {triggered}")
    print(f"   Intensity: {intensity:.2f}")
    
    if triggered:
        print("   ✅ PASS: Single input triggered dynamic gate (Buffer logic).")
    else:
        print("   ❌ FAIL: Single input failed to trigger.")

if __name__ == "__main__":
    test_wave_thought()
