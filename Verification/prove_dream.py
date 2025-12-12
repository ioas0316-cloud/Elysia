"""
Prove Dream (상상 증명)
=====================

엘리시아가 기억을 재조합하여 꿈을 꾸는지 검증합니다.
"""

from Core.Cognitive.memory_stream import get_memory_stream, ExperienceType
from Core.Cognitive.dream_integrator import get_dream_integrator

def prove_dream():
    print("💤 Dream Verification Started...\n")
    
    memory = get_memory_stream()
    integrator = get_dream_integrator()
    
    # 1. Seed Memories (Create fake past experiences)
    print("1. Planting memories...")
    memory.add_experience(ExperienceType.CREATION, 
                         {"intent": "Ocean"}, 
                         {"content": "Blue waves crashing"}, 
                         {"description": "Vast and deep"})
                         
    memory.add_experience(ExperienceType.CREATION, 
                         {"intent": "Fire"}, 
                         {"content": "Red flames dancing"}, 
                         {"description": "Hot and bright"})
    
    # 2. Dream Walk
    print("\n2. Walking into the Dream (Recombination)...")
    dream_exp = integrator.dream_walk()
    
    # 3. Result
    if dream_exp:
        print(f"   ✨ Dream weaved: {dream_exp.score['intent']}")
        print(f"   🧩 Insight: {dream_exp.sound['insight']}")
        print(f"   🌀 Internal Mix: {dream_exp.sound['surreal_mix']}")
        print("\n✅ SUCCESS: Memories were recombined into a dream.")
    else:
        print("\n❌ FAIL: Dream was not created.")

if __name__ == "__main__":
    prove_dream()
