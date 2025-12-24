
"""
Verify Field Causality (Phase 25)
=================================

Tests the "Cloud & Lightning" logic.
Ensures that concepts only connect (Spark) when:
1. Tension is high (Charge > Threshold)
2. Shapes are compatible (Causal Fit)
"""

import sys
import os
sys.path.append("c:\\Elysia")

from Core.02_Intelligence.01_Reasoning.Cognition.Reasoning.causal_geometry import TensionField, CausalShape

def test_causal_fit():
    print("🧪 Testing Causal Fit Logic...")
    
    field = TensionField(threshold=0.8)
    
    # 1. Define Concepts
    field.register_concept("Question", auto_shape=False)
    field.register_concept("Answer", auto_shape=False)
    field.register_concept("Noise", auto_shape=False)
    
    # 2. Define Shapes (Puzzle Pieces)
    # Question: Needs Logic (+)
    q_shape = field.shapes["Question"]
    q_shape.add_port("Logic", polarity=1) # Output: Logic Request
    
    # Answer: Provides Logic (-) -> Fits Question
    a_shape = field.shapes["Answer"]
    a_shape.add_port("Logic", polarity=-1) # Input: Logic Provider
    
    # Noise: Emotion (+) -> Does NOT fit Logic
    n_shape = field.shapes["Noise"]
    n_shape.add_port("Emotion", polarity=1)
    
    print("   🧩 Shapes Defined:")
    print(f"      Question: {q_shape.ports}")
    print(f"      Answer:   {a_shape.ports}")
    print(f"      Noise:    {n_shape.ports}")
    
    # 3. Test Low Tension (Should fail)
    print("\n   ☁️ Test 1: Low Tension...")
    field.charge_concept("Question", 0.3)
    field.charge_concept("Answer", 0.3)
    sparks = field.discharge_lightning()
    if not sparks:
        print("      ✅ Passed: No lightning at low tension.")
    else:
        print(f"      ❌ Failed: Lightning struck prematurely! {sparks}")

    # 4. Test High Tension but Wrong Shape (Should fail)
    print("\n   ☁️ Test 2: High Tension / Wrong Shape...")
    field.charge_concept("Question", 0.9)
    field.charge_concept("Noise", 0.9)
    # Reset Answer charge for isolation
    field.charges["Answer"] = 0.0 
    
    sparks = field.discharge_lightning()
    if not sparks:
        print("      ✅ Passed: No lightning for mismatched shapes (Logic vs Emotion).")
    else:
        print(f"      ❌ Failed: Lightning struck mismatched shapes! {sparks}")

    # 5. Test High Tension & Correct Shape (Should SPARK)
    print("\n   ⚡ Test 3: High Tension / Correct Shape...")
    field.charge_concept("Question", 0.9) # Still high
    field.charge_concept("Answer", 0.9)   # Boost Answer
    
    sparks = field.discharge_lightning()
    if sparks:
        print(f"      ✅ Passed: LIGHTNING STRUCK! {sparks[0][0]} <==> {sparks[0][1]}")
    else:
        print("      ❌ Failed: No lightning for perfect conditions!")

if __name__ == "__main__":
    try:
        test_causal_fit()
        print("\n✨ Verification Complete.")
    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()
