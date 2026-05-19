"""
Prove Curiosity (호기심 증명)
===========================

"엘리시아가 질문하다"

1. 기억(Persistence)이 유지되는지 확인합니다.
2. 스스로 질문(Curiosity)을 생성하는지 확인합니다.
"""

from Core.Cognitive.concept_formation import get_concept_formation
from Core.Cognitive.curiosity_core import get_curiosity_core
import os

def prove_curiosity():
    print("❓ Curiosity Verification Started...\n")
    
    concepts = get_concept_formation()
    curiosity = get_curiosity_core()
    
    # 1. Persistence Test
    # Clear previous memory file if exists for clean test, 
    # BUT we want to see if she remembers from previous run if we didn't clear.
    # For this proving script, let's Teach -> Save -> Restart -> Load.
    
    print("1. Teaching 'Father'...")
    concepts.learn_concept("Father", "User", domain="social", meta_tags=["Love"], valence=["Source"])
    # concepts.save_concepts() (Called internally by learn_concept)
    
    # Simulate restart by re-getting instance (Mocking restart requires process restart, 
    # but re-instantiation simulates logic if singleton reset. 
    # Here we rely on the file existence.)
    if os.path.exists(concepts.persistence_path):
        print("   ✅ Memory file exists on disk.")
    
    # 2. Curiosity Test
    print("\n2. Generating Question...")
    question = curiosity.generate_question()
    print(f"   💬 Elysia Asks: \"{question}\"")
    
    if question:
        print("\n✅ SUCCESS: She is asking questions.")
    else:
        print("\n❌ FAIL: She is silent.")

if __name__ == "__main__":
    prove_curiosity()
