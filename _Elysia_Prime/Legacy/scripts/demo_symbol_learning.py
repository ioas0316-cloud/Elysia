# [Genesis: 2025-12-02] Purified by Elysia
"""
Demo: symbol (문자/단어) learning episodes driven by WORLD characters.

Design
- WORLD / Character는 "학습자" 역할을 한다.
- 이 스크립트는 임의의 한글/영어 기호에 대해 퀴즈를 내고
  각 캐릭터가 맞추거나 틀리는 과정을 SymbolEpisode 원시 로그로 기록한다.
- 학습/정답 여부는 per-character, per-symbol skill 값으로 결정되며,
  정답일수록 skill이 조금씩 올라가는 단순 인과 구조를 따른다.

Output
- logs/symbol_lessons.jsonl : raw lesson events (SYMBOL_LESSON ...)
- logs/symbol_episodes.jsonl : 정규화된 SymbolEpisode JSONL
- 터미널 : 기호 종류/기호별 정확도 리포트
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple


def _ensure_repo_root_on_path() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


_ensure_repo_root_on_path()


from scripts.export_world_snapshot_for_godot import _build_world  # noqa: E402
from scripts.causal_episodes import (  # noqa: E402
    export_symbol_episodes,
    print_symbol_learning_report,
)


# --- Symbol universe ---------------------------------------------------------

HANGUL_JAMO: List[str] = list("ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ")
HANGUL_SYLLABLES: List[str] = ["가", "나", "다", "라", "마", "바", "사", "아", "자", "차"]
EN_LETTERS: List[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EN_WORDS: List[str] = ["king", "queen", "sword", "village", "dragon"]


def _pick_symbol() -> Tuple[str, str]:
    """
    Return (symbol_type, symbol) pair.
    """
    bucket = random.random()
    if bucket < 0.25:
        return "letter_ko", random.choice(HANGUL_JAMO)
    if bucket < 0.5:
        return "syllable_ko", random.choice(HANGUL_SYLLABLES)
    if bucket < 0.75:
        return "letter_en", random.choice(EN_LETTERS)
    return "word_en", random.choice(EN_WORDS)


def _simulate_lesson_for_character(
    char_id: str,
    timestamp: int,
    skill_table: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Simulate one SYMBOL_LESSON event for a single character.

    - skill_table[char_id][symbol] in [0,1] controls 정답 확률.
    - 정답이면 skill += 0.1, 오답이면 skill -= 0.05 (0~1 클램프).
    """
    symbol_type, symbol = _pick_symbol()

    per_char = skill_table.setdefault(char_id, {})
    key = f"{symbol_type}:{symbol}"
    skill = per_char.get(key, 0.2)  # 기본 초보 수준

    # 정답 확률 = 현재 skill
    p_correct = max(0.0, min(1.0, skill))
    roll = random.random()
    correct = roll < p_correct

    if correct:
        learner_guess = symbol
        score = 1.0
        skill = min(1.0, skill + 0.1)
    else:
        # 오답 시: 같은 종류에서 다른 기호를 하나 고른다.
        if symbol_type == "letter_ko":
            pool = [s for s in HANGUL_JAMO if s != symbol]
        elif symbol_type == "syllable_ko":
            pool = [s for s in HANGUL_SYLLABLES if s != symbol]
        elif symbol_type == "letter_en":
            pool = [s for s in EN_LETTERS if s != symbol]
        else:
            pool = [s for s in EN_WORDS if s != symbol]
        learner_guess = random.choice(pool) if pool else ""
        score = 0.0
        skill = max(0.0, skill - 0.05)

    per_char[key] = skill

    event = {
        "timestamp": timestamp,
        "event_type": "SYMBOL_LESSON",
        "data": {
            "symbol_type": symbol_type,
            "symbol": symbol,
            "modality": "text",
            "teacher_label": symbol,
            "learner_guess": learner_guess,
            "correct": correct,
            "score": score,
            "learner_id": char_id,
        },
    }
    return event


def run_symbol_learning_demo(
    lessons_per_char: int = 50,
    seed: int = 42,
) -> None:
    """
    Run a small symbol learning simulation driven by WORLD characters.

    - For each character in the sample world, run `lessons_per_char` lessons.
    - Log results to `logs/symbol_lessons.jsonl`.
    - Convert to SymbolEpisode and print a summary report.
    """
    random.seed(seed)

    world, chars, _macro_states = _build_world(years=50, ticks_per_year=2)

    # 학습자는 전체 캐릭터 중 일부 샘플만 사용.
    learners = [ch for ch in chars if ch.id and ch.name]
    if not learners:
        print("[symbol_demo] No learners available.")
        return

    logs_dir = os.path.join(_ensure_repo_root_on_path(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    lessons_path = os.path.join(logs_dir, "symbol_lessons.jsonl")

    skill_table: Dict[str, Dict[str, float]] = {}
    timestamp = 0

    with open(lessons_path, "w", encoding="utf-8") as f:
        for ch in learners:
            for _ in range(lessons_per_char):
                ev = _simulate_lesson_for_character(ch.id, timestamp, skill_table)
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                timestamp += 1

    print(f"[symbol_demo] Wrote raw lessons to: {lessons_path}")

    # Convert to SymbolEpisode + print report.
    symbol_episodes_path = os.path.join(logs_dir, "symbol_episodes.jsonl")
    export_symbol_episodes(events_path=lessons_path, out_path=symbol_episodes_path)
    print(f"[symbol_demo] Wrote symbol episodes to: {symbol_episodes_path}")

    print_symbol_learning_report(symbol_episodes_path)


if __name__ == "__main__":
    run_symbol_learning_demo()
