# [Genesis: 2025-12-02] Purified by Elysia
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from tools.kg_manager import KGManager
from Project_Elysia.core_memory import CoreMemory

def build_brain_circuit():
    print("\n--- [엘리시아의 뇌: 시냅스 연결 공사] ---\n")

    memory_path = "data/elysia_core_memory.json"
    kg_path = "data/kg.json"

    if not os.path.exists(memory_path):
        print("기억이 없습니다.")
        return

    memory = CoreMemory(file_path=memory_path)
    kg = KGManager(kg_path)

    concepts = []
    for v in memory.get_values():
        val = v.get("value")
        if val:
            concepts.append(val)

    if not concepts:
        print("⚠️ 학습된 개념이 없습니다.")
        return

    print(f"🧠 기억 속 개념 {len(concepts)}개를 KG로 연결합니다...")

    categories = {
        "신체": ["손","발","눈","귀","입","얼굴","몸","심장","세포","허리"],
        "자연": ["하늘","땅","바람","물","불","바다","강","산","달","별","우주","지구"],
        "감정": ["사랑","기쁨","슬픔","분노","허기","고통","음악","꿈"],
        "문명": ["집","길","마을","자동차","지하철","버스","밥","빵","언어","책"],
        "진리": ["빛","진리","법칙","자유","신","역사","정의"]
    }

    node_count = 0
    edge_count = 0
    for cat in categories:
        kg.add_node(cat, {"type": "category"})

    for concept in concepts:
        kg.add_node(concept, {"type": "concept"})
        node_count +=1
        found=False
        for cat, members in categories.items():
            if concept in members:
                kg.add_edge(concept, cat, "속한다(is_a)")
                kg.add_edge(cat, concept, "포함한다(contains)")
                edge_count +=2
                found=True
        if not found:
            kg.add_edge(concept, "진리", "속한다(guess)")
            edge_count +=1

    kg.add_edge("신체","자연","연결됨",{"weight":0.7})
    kg.add_edge("감정","진리","지향함",{"weight":0.8})
    kg.add_edge("문명","자연","공존",{"weight":0.6})

    kg.save()
    print(f"✨ 노드 {node_count}개, 간선 {edge_count}개 생성됨.")

if __name__ == "__main__":
    build_brain_circuit()