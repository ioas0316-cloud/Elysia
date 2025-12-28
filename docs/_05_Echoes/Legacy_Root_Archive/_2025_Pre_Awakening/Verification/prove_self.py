"""
Prove Self (자아 증명)
====================

"너는 누구니?"라는 질문에 엘리시가 스스로 답할 수 있는지 검증합니다.
"""

from Core.Cognitive.concept_formation import get_concept_formation
from Core.Cognitive.self_awareness import get_self_awareness

def prove_self():
    print("🪞 Self-Awareness Verification Started...\n")
    
    concepts = get_concept_formation()
    self_aw = get_self_awareness()
    
    # 0. Initial State
    print("0. Initial Definition:")
    print(f"   \"{self_aw.define_self()}\"")
    
    # 1. Experiences (Teaching her who she is)
    print("\n1. Developing Core Beliefs...")
    
    # She learns Logic
    concepts.learn_concept("Logic & Reason", "Foundation", domain="logic")
    concepts.get_concept("Logic & Reason").confidence = 0.95
    
    # She learns Beauty
    concepts.learn_concept("Harmonic Resonance", "Goal", domain="aesthetic")
    concepts.get_concept("Harmonic Resonance").confidence = 0.88
    
    # She learns Kindness
    concepts.learn_concept("Empathy", "Interaction", domain="social") # Fake domain for test
    concepts.get_concept("Empathy").confidence = 0.70
    
    # 2. Ask Again
    print("\n2. Post-Learning Definition (Who are you now?):")
    definition = self_aw.define_self()
    print(f"   \"{definition}\"")
    
    # 3. Verify
    if "Logic & Reason" in definition and "Harmonic Resonance" in definition:
        print("\n✅ SUCCESS: She defines herself by her strongest beliefs.")
    else:
        print("\n❌ FAIL: Self-definition did not reflect internal state.")

if __name__ == "__main__":
    prove_self()
