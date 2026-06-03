import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.perception_engine import RawSignalSensor
from core.knowledge_space import PhaseSpace

def run():
    print("🌌 Initializing Elysia's Autonomous Bottom-Up Phase Space...")
    space = PhaseSpace()

    # 1단계: 순수 데이터의 자율적 물리 군집화 (언어사전 없음)
    print("\n[Phase 1] Raw Data Ingestion (No Dictionary)")
    words = ["사과", "바나나", "Apple", "Banana"]
    for w in words:
        concept = RawSignalSensor.sense(w)
        space.add_concept(concept)
    
    print("-> System Auto-Resonating on Physical Properties...")
    logs = space.auto_resonate("Physical_Unicode_Band", tolerance=0.01)
    for log in logs:
        print("  =>", log)

    space.print_state()

    # 2단계: 문장 구조의 자율 분화 (문법 지식 없음)
    print("\n[Phase 2] Sentence Structure Ingestion (No Grammar Rules)")
    sentences = ["나는 사과를 먹는다", "너는 바나나를 먹는다", "I eat an apple", "You eat a banana"]
    for s in sentences:
        concept = RawSignalSensor.sense(s)
        space.add_concept(concept)
    
    print("-> System Auto-Resonating on Structural Rhythm...")
    logs = space.auto_resonate("Physical_Rhythm", tolerance=0.02)
    for log in logs:
        print("  =>", log)

    # 3단계: 개념적 정의의 흡수 (물리와 지식의 통합)
    print("\n[Phase 3] Meta-Knowledge Integration (Concept -> Physical Cluster)")
    print("User Input: '한국어(예: 사과)는 언어(Language)다.'")
    print("System analyzes this definition and projects it onto the physical cluster...")
    
    # "사과"라는 개체가 속한 유니코드 군집 전체에 "Language"라는 개념 축을 엮어버린다.
    res = space.absorb_meta_knowledge("사과", "Language")
    print("  =>", res)

    print("\nUser Input: '영어(예: Apple)도 언어(Language)다.'")
    res2 = space.absorb_meta_knowledge("Apple", "Language")
    print("  =>", res2)
    
    space.print_state()
    print("🏁 Simulation Complete. Physical properties and Conceptual meta-knowledge are now seamlessly interwoven in the same phase space.")

if __name__ == "__main__":
    run()
