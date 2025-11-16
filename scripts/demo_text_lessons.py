"""
Demo: sentence/paragraph lessons based on WORLD narratives.

Purpose
- Use existing narrative helpers (era flags, character arcs) to create
  teacher_text 문장들을 만들고,
- 이를 SENTENCE_LESSON 이벤트로 기록하여 TextEpisode 학습 재료를 만든다.
- 동시에, 이 문장들을 WORLD 안 "텍스트 오브젝트(책/일기)"로 심어
  Godot 스냅샷에 노출할 수 있게 한다.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List


def _ensure_repo_root_on_path() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


_ensure_repo_root_on_path()

from scripts.export_world_snapshot_for_godot import _build_world  # noqa: E402
from scripts.narrative_summaries import summarize_era_flags, summarize_character_arc  # noqa: E402
from scripts.causal_episodes import export_text_episodes  # noqa: E402


def run_text_lessons_demo(
    seed: int = 123,
    max_char_lessons: int = 10,
) -> None:
    repo_root = _ensure_repo_root_on_path()
    logs_dir = os.path.join(repo_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    world, chars, _macro_states = _build_world(years=300, ticks_per_year=3)

    lessons_path = os.path.join(logs_dir, "text_lessons.jsonl")
    text_objects_path = os.path.join(logs_dir, "world_text_objects.json")

    events: List[Dict[str, Any]] = []
    text_objects: List[Dict[str, Any]] = []
    ts = 0

    # --- 1) Era summary sentences as lessons + world-level "books" ----------

    era_lines = summarize_era_flags(world)
    for i, line in enumerate(era_lines):
        ev = {
            "timestamp": ts,
            "event_type": "SENTENCE_LESSON",
            "data": {
                "text_type": "sentence_ko",
                "teacher_text": line,
                "learner_text": line,
                "correct": True,
                "score": 1.0,
                "scope": "world_era",
                "lesson_id": f"era_{i}",
            },
        }
        events.append(ev)
        ts += 1

        text_objects.append(
            {
                "id": f"book_era_{i}",
                "kind": "book",
                "owner_id": None,
                "title_ko": "시대 연대기",
                "text_ko": line,
                "position": None,
            }
        )

    # --- 2) Character arc sentences as personal diaries ---------------------

    # 간단히 상위 N명만 사용.
    used = 0
    for ch in chars:
        if used >= max_char_lessons:
            break
        arc = summarize_character_arc(ch)
        ev = {
            "timestamp": ts,
            "event_type": "SENTENCE_LESSON",
            "data": {
                "text_type": "sentence_ko",
                "teacher_text": arc,
                "learner_text": arc,
                "correct": True,
                "score": 1.0,
                "scope": "character_arc",
                "character_id": ch.id,
            },
        }
        events.append(ev)
        ts += 1

        idx = world.id_to_idx.get(ch.id)
        position = None
        try:
            if idx is not None and world.position is not None and idx < world.position.shape[0]:
                pos_vec = world.position[idx]
                position = {
                    "x": float(pos_vec[0]),
                    "y": float(pos_vec[1]),
                    "z": float(pos_vec[2]) if len(pos_vec) > 2 else 0.0,
                }
        except Exception:
            position = None

        text_objects.append(
            {
                "id": f"diary_{ch.id}",
                "kind": "diary",
                "owner_id": ch.id,
                "title_ko": "인생의 한 줄",
                "text_ko": arc,
                "position": position,
            }
        )
        used += 1

    # --- 3) Write raw lesson events ----------------------------------------

    with open(lessons_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    print(f"[text_demo] Wrote text lessons to: {lessons_path}")

    # --- 4) Export TextEpisode JSONL ----------------------------------------

    text_episodes_path = os.path.join(logs_dir, "text_episodes.jsonl")
    export_text_episodes(events_path=lessons_path, out_path=text_episodes_path)
    print(f"[text_demo] Wrote text episodes to: {text_episodes_path}")

    # --- 5) Write world text objects overlay --------------------------------

    with open(text_objects_path, "w", encoding="utf-8") as f:
        json.dump({"texts": text_objects}, f, ensure_ascii=False, indent=2)
    print(f"[text_demo] Wrote world text objects to: {text_objects_path}")


if __name__ == "__main__":
    run_text_lessons_demo()

