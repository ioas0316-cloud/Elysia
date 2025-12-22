"""
Prove Hologram (홀로그램 증명)
============================

"고양이의 수염만 보고 고양이를 알다"

부분적인 정보(70%)만 주어졌을 때, 
나머지를 상상(Imagination)하여 전체를 인식하는지 검증합니다.
"""

from Core.Cognitive.concept_formation import get_concept_formation
from Core.Cognitive.holographic_cortex import get_holographic_cortex

def prove_hologram():
    print("🔮 Holographic Inference Verification Started...\n")
    
    concepts = get_concept_formation()
    hologram = get_holographic_cortex()
    
    # 1. Teach the Whole (The Ideal Form)
    print("1. Learning the Concept 'Cat'...")
    concepts.learn_concept(
        name="Cat", 
        context="Animal", 
        domain="nature",
        meta_tags=["Whiskers", "PointedEars", "Tail", "Meow", "Fur"]
    )
    
    # 2. Present Partial Data (The Whiskers)
    print("\n2. Observing Partial Features: ['Whiskers', 'Meow']")
    observation = ["Whiskers", "Meow"]
    
    # 3. Reconstruct
    result = hologram.reconstruct(observation)
    
    # 4. Verify
    if result and result["concept"] == "Cat":
        print(f"\n✅ SUCCESS: Identified '{result['concept']}' from incomplete data.")
        print(f"   Imagined Features: {result['imagined']}")
    else:
        print("\n❌ FAIL: Could not complete the pattern.")

if __name__ == "__main__":
    prove_hologram()
