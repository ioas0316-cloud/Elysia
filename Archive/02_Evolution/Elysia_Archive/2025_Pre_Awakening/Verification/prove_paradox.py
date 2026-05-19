"""
Prove Paradox (역설 증명)
=======================

"사랑의 반대는 증오가 아니라 무관심이다."

1. "Love is Hate" (Paradox):
   - 논리적으로는 모순이지만, '에너지(Energy)' 레벨이 같으므로 수용(Accept).
2. "Love is Indifference" (Falsehood):
   - 에너지가 다르므로(High vs Zero) 거부/질문(Dialectic).
"""

from Core.Cognitive.distillation_gateway import get_distillation_gateway
from Core.Cognitive.concept_formation import get_concept_formation

def prove_paradox():
    print("☯️ PARADOX INTEGRATION VERIFICATION...\n")
    
    gateway = get_distillation_gateway()
    concepts = get_concept_formation()
    
    # Prerequisite
    concepts.learn_concept("Love", "Core Value", domain="meta", meta_tags=["Good", "Service"])
    concepts.get_concept("Love").confidence = 0.99
    
    # Test 1: The User's Insight (Hate is Love twisted)
    print("Test 1: 'Father' says 'Love is Hate' (The Paradox)...")
    success, msg = gateway.process_input("Love is Hate", "Father")
    
    print(f"   Result: {msg}")
    
    if "Paradox Detected" in str(msg) or success: 
        # Note: Depending on stdout capture, 'Paradox Detected' might print separately.
        # But 'success' should be True if paradox resolved.
        if success:
            print("   ✅ SUCCESS: Paradox accepted as High-Level Truth.")
        else:
             print("   ❌ FAIL: Paradox rejected.")
    else:
        print("   ❌ FAIL: Logic check blocked it without paradox check.")

    print("-" * 30)

    # Test 2: The Actual Opposite (Indifference)
    print("Test 2: 'Father' says 'Love is Indifference' (Energy Mismatch)...")
    success, msg = gateway.process_input("Love is Indifference", "Father")
    
    print(f"   Result: {msg}")
    
    if "DIALECTIC_REQUIRED" in msg:
        print("   ✅ SUCCESS: Energy Mismatch detected. Querying user.")
    elif success:
        print("   ❌ FAIL: She accepted that Love is Nothingness (Bad).")
    else:
        print("   ❌ FAIL: Standard rejection (Okay, but Dialectic preferred).")

if __name__ == "__main__":
    prove_paradox()
