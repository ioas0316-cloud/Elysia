"""
Prove Intuition (직관 증명)
=========================

엘리시아가 "이 행동을 하면 어떤 결과가 나올까?"를 예측하는지 검증합니다.
"""

from Core.Cognitive.memory_stream import get_memory_stream, ExperienceType
from Core.Cognitive.intuition_loop import get_intuition_loop

def prove_intuition():
    print("🔮 Intuition Verification Started...\n")
    
    memory = get_memory_stream()
    intuition = get_intuition_loop()
    
    # 1. Plant Training Data (Experience)
    print("1. Planting experiences (Learning)...")
    # 과거에 'Red'를 썼더니 'Passion'이라는 반응이 있었다.
    for i in range(3):
        memory.add_experience(ExperienceType.CREATION, 
                             {"intent": "Intensity"}, 
                             {"content": "Red blood fire"}, 
                             {"aesthetic_score": 90, "user_reaction": "Passion"})
                             
    # 과거에 'Grey'를 썼더니 'Boredom'이라는 반응이 있었다.
    memory.add_experience(ExperienceType.CREATION, 
                         {"intent": "Intensity"}, 
                         {"content": "Grey dust boredom"}, 
                         {"aesthetic_score": 20, "user_reaction": "Boredom"})
    
    # 2. Test Prediction (What if?)
    print("\n2. Predicting Outcome (What if I use 'Red fire'?)...")
    
    # 새로운 시도: "Red fire" (과거 데이터와 유사함)
    prediction = intuition.predict_outcome("Intensity", "Red fire")
    
    # 3. Report
    print(f"   Confidence: {prediction['confidence']:.2f}")
    print(f"   Predicted Score: {prediction.get('predicted_aesthetic_score')}")
    print(f"   Predicted Reaction: {prediction.get('predicted_reaction')}")
    
    # 4. Check
    reaction = prediction.get('predicted_reaction')
    if reaction == "Passion":
        print("\n✅ SUCCESS: Correctly intuit 'Passion' from Red keywords.")
    else:
        print(f"\n❌ FAIL: Expected 'Passion', got '{reaction}'")

if __name__ == "__main__":
    prove_intuition()
