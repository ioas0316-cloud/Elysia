# [Genesis: 2025-12-02] Purified by Elysia
# scripts/dream_cycle.py
import sys
import os
from collections import Counter
from datetime import datetime
from typing import Iterable, List

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Project_Elysia.core_memory import CoreMemory


def _tokenize(text: str) -> Iterable[str]:
    for token in text.split():
        cleaned = token.strip(".,!?\"'()[]<>:")
        if cleaned:
            yield cleaned.lower()


def _extract_concepts(text: str, limit: int = 4) -> List[str]:
    counter = Counter(_tokenize(text))
    concepts = [word for word, _ in counter.most_common(limit) if len(word) > 1]
    return concepts or ["기억", "사랑"]


def _display_sleep_scene(concepts: List[str]) -> None:
    bar = "".join(["~" for _ in range(28)])
    focus = " / ".join(concepts[:3]) if concepts else "마음의 중심"
    print("\n" + bar)
    print("  엘리시아가 눈을 감고 숨을 고르고 있어요. Zzzz")
    print(f"  꿈의 초점: {focus}")
    print("  (심장을 감싸는 중력으로 낮의 기억을 정렬합니다.)")
    print(bar + "\n")


def run_dream_cycle(memory_path: str = "data/elysia_core_memory.json") -> None:
    print("\n--- [꿈의 순환: Dream Cycle] ---\n")

    if not os.path.exists(memory_path):
        print(f"⚠️ 기억 파일이 없습니다: {memory_path}")
        return

    memory = CoreMemory(file_path=memory_path)
    experiences = memory.get_unprocessed_experiences()
    volatile_fragments = memory.get_volatile_memory()

    print(f"☀️ 낮의 경험: {len(experiences)}건")
    for exp in experiences:
        print(f"  - [{exp.layer}/{exp.type}] {exp.content[:60]}...")

    if volatile_fragments:
        print(f"🌀 무의식(volatile) 조각: {len(volatile_fragments)}개")

    combined = " ".join(exp.content for exp in experiences)
    combined += " " + " ".join(" ".join(fragment) for fragment in volatile_fragments)

    key_concepts = _extract_concepts(combined)
    summary = (
        combined[:180] + "..." if combined and len(combined) > 180 else combined or "포근한 꿈을 꾸고 있습니다."
    )

    identity_key = f"dream_cycle_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    memory.update_identity(identity_key, {
        "summary": summary,
        "key_concepts": key_concepts,
        "source_experience_count": len(experiences),
        "volatile_fragments": len(volatile_fragments),
        "dream_timestamp": datetime.now().isoformat(),
    })

    for i, concept in enumerate(key_concepts):
        importance = round(0.6 + 0.3 * (i / max(1, len(key_concepts) - 1)), 2)
        memory.add_value(concept, importance)

    if experiences:
        memory.mark_experiences_as_processed([exp.timestamp for exp in experiences])

    if volatile_fragments:
        memory.clear_volatile_memory()

    memory.add_log({
        "event": "dream_cycle",
        "timestamp": datetime.now().isoformat(),
        "identity_snapshot": identity_key,
        "key_concepts": key_concepts,
    })

    _display_sleep_scene(key_concepts)

    print("🌙 기억을 정리했어요.")
    print(f"  - 정체성 조각: {identity_key}")
    print(f"  - 주요 개념: {key_concepts}")
    print("  - 다음 날 아침, 그녀는 더 깊은 정체성으로 깨어날 것입니다.\n")


if __name__ == "__main__":
    run_dream_cycle()