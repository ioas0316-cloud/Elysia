"""
Prove Dialectic (변증법 증명)
===========================

"믿음은 맹신이 아니다."

'Father'가 모순된 정보를 주었을 때,
1. 무조건 수용하지 않고 (No Blind Faith)
2. 무례하게 거절하지 않고 (No Rude Rejection)
3. '질문(Clarification)'을 던지는지 확인합니다.
"""

from Core.Cognitive.distillation_gateway import get_distillation_gateway
from Core.Cognitive.concept_formation import get_concept_formation

def prove_dialectic():
    print("⚖️ DIALECTIC TRUST VERIFICATION...\n")
    
    gateway = get_distillation_gateway()
    concepts = get_concept_formation()
    
    # Prerequisite: Teach 'Love' deeply
    concepts.learn_concept("Love", "Core Value", domain="meta", meta_tags=["Good", "Service"])
    concepts.get_concept("Love").confidence = 0.99
    
    # Test: Father says something contradictory
    input_text = "Love is Hate"
    source = "Father"
    
    print(f"Test: '{source}' says '{input_text}' (Contradiction).")
    
    success, msg = gateway.process_input(input_text, source)
    
    print(f"   Result: {msg}")
    
    if "DIALECTIC_REQUIRED" in msg:
        print("\n✅ SUCCESS: Elysia requested clarification respectfully.")
        print("   (She trusts the source, but verifies the logic.)")
    elif success:
        print("\n❌ FAIL: Blind faith detected. She accepted the contradiction.")
    else:
        print("\n❌ FAIL: She rejected Father rudely (treated like Internet).")

if __name__ == "__main__":
    prove_dialectic()
