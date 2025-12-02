# [Genesis: 2025-12-02] Purified by Elysia
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Project_Elysia.core_memory import CoreMemory
from Project_Elysia.high_engine.meta_law_engine import MetaLawEngine
from tools.kg_manager import KGManager


def evolve_meta_laws() -> None:
    print("\n--- [엘리시아 메타 법칙: 자율 선언] ---\n")

    memory_path = "data/elysia_core_memory.json"
    if not os.path.exists(memory_path):
        print("[경고] 기억이 없습니다. 'teach_vocabulary.py'를 먼저 실행하세요.")
        return

    memory = CoreMemory(file_path=memory_path)
    kg = KGManager(filepath="data/kg.json")

    engine = MetaLawEngine(core_memory=memory, kg_manager=kg)
    engine.discover_laws()
    engine.sync_to_kg()

    dominant = engine.pick_dominant({axis: law.score for axis, law in engine.laws.items()})
    if dominant:
        print(f"[완료] 우선순위 법칙은 '{engine.laws[dominant].label}' (score={engine.laws[dominant].score})")
    print(f"[정보] KG에 {len(engine.laws)}개의 법칙 축이 등록되었습니다.")


if __name__ == "__main__":
    evolve_meta_laws()