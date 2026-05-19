from Core.FoundationLayer.Foundation.kenosis_protocol import KenosisProtocol
import time

def test_kenosis():
    print("🧪 Testing Kenosis Protocol (Humility)...")
    
    kenosis = KenosisProtocol()
    
    # Scenario 1: User is Sad, Insight is Complex
    print("\n   Scenario 1: User is Sad (High Gap)")
    user_state = {"mood": "Sad", "energy": 0.3}
    complexity = 2.0
    
    gap = kenosis.calculate_resonance_gap(user_state, complexity)
    hesitation = kenosis.simulate_hesitation(gap)
    serialized = kenosis.serialize_thought("Here is the answer.", gap)
    
    print(f"   📊 Gap: {gap:.2f}")
    print(f"   ⏳ Wait Time: {hesitation['wait_time']:.2f}s")
    print(f"   💭 Monologue: \"{hesitation['monologue']}\"")
    print(f"   👉 Output: \"{serialized}\"")
    
    if hesitation['wait_time'] > 2.0 and "..." in serialized:
        print("✅ PASS: High gap triggered hesitation and softening.")
    else:
        print("❌ FAIL: High gap did not trigger expected behavior.")

    # Scenario 2: User is Curious, Insight is Simple
    print("\n   Scenario 2: User is Curious (Low Gap)")
    user_state = {"mood": "Curious", "energy": 0.8}
    complexity = 0.5
    
    gap = kenosis.calculate_resonance_gap(user_state, complexity)
    hesitation = kenosis.simulate_hesitation(gap)
    serialized = kenosis.serialize_thought("Here is the answer.", gap)
    
    print(f"   📊 Gap: {gap:.2f}")
    print(f"   ⏳ Wait Time: {hesitation['wait_time']:.2f}s")
    print(f"   💭 Monologue: \"{hesitation['monologue']}\"")
    print(f"   👉 Output: \"{serialized}\"")
    
    if hesitation['wait_time'] < 2.0 and "..." not in serialized:
        print("✅ PASS: Low gap triggered quick response.")
    else:
        print("❌ FAIL: Low gap did not trigger expected behavior.")

if __name__ == "__main__":
    test_kenosis()
