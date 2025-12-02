# [Genesis: 2025-12-02] Purified by Elysia
"""
Demo: use learned symbols to attach simple "signs" to WORLD characters.

Design
- Read WORLD snapshot by rebuilding a small sample world.
- Read symbol learning episodes and approximate 각 캐릭터의 문자 지식.
- For literate characters, create simple sign texts (한글/영어) and
  export them as an overlay JSON for Godot/Elysia to visualize.

This does not change WORLD physics; it only emits:
- logs/world_symbol_usage.json
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
from scripts.causal_episodes import summarize_symbol_episodes_dicts  # noqa: E402


def _load_symbol_episodes(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    episodes: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return episodes


def _compute_per_char_knowledge(
    episodes: List[Dict[str, Any]],
    min_accuracy: float = 0.6,
    min_count: int = 5,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Return mapping: learner_id -> {symbol_type: [known symbols]}.
    """
    per_char_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    for ep in episodes:
        data = ep.get("data", {}) or {}
        learner_id = data.get("learner_id")
        if not learner_id:
            continue
        stype = str(data.get("symbol_type") or "unknown")
        symbol = str(data.get("symbol") or "")
        correct = bool(data.get("correct", False))

        bucket = per_char_stats.setdefault(learner_id, {})
        stats = bucket.setdefault(symbol, {"count": 0.0, "correct": 0.0, "symbol_type": stype})
        stats["count"] += 1.0
        if correct:
            stats["correct"] += 1.0

    per_char_known: Dict[str, Dict[str, List[str]]] = {}
    for learner_id, sym_stats in per_char_stats.items():
        type_map: Dict[str, List[str]] = {}
        for symbol, stats in sym_stats.items():
            count = stats["count"]
            correct = stats["correct"]
            if count < min_count:
                continue
            acc = correct / count if count > 0 else 0.0
            if acc < min_accuracy:
                continue
            stype = str(stats["symbol_type"])
            type_map.setdefault(stype, []).append(symbol)
        if type_map:
            per_char_known[learner_id] = type_map

    return per_char_known


def run_world_usage_demo() -> None:
    repo_root = _ensure_repo_root_on_path()
    logs_dir = os.path.join(repo_root, "logs")
    symbol_episodes_path = os.path.join(logs_dir, "symbol_lessons.jsonl")

    episodes = _load_symbol_episodes(symbol_episodes_path)
    if not episodes:
        print(f"[world_symbol_usage] No symbol lessons at {symbol_episodes_path}")
        return

    per_char_known = _compute_per_char_knowledge(episodes)

    # Build a world snapshot so we can attach positions.
    world, chars, _macro_states = _build_world(years=50, ticks_per_year=2)

    signs: List[Dict[str, Any]] = []
    for ch in chars:
        kv = per_char_known.get(ch.id)
        if not kv:
            continue

        # Choose simple texts from known symbols.
        letters_ko = kv.get("letter_ko", [])
        syllables_ko = kv.get("syllable_ko", [])
        letters_en = kv.get("letter_en", [])
        words_en = kv.get("word_en", [])

        text_ko = ""
        text_en = ""

        if syllables_ko:
            text_ko = "".join(syllables_ko[:3])
        elif letters_ko:
            text_ko = "".join(letters_ko[:3])

        if words_en:
            text_en = " ".join(words_en[:2])
        elif letters_en:
            text_en = "".join(letters_en[:3])

        if not text_ko and not text_en:
            continue

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

        signs.append(
            {
                "owner_id": ch.id,
                "owner_name": ch.name,
                "text_ko": text_ko or None,
                "text_en": text_en or None,
                "position": position,
            }
        )

    out_path = os.path.join(logs_dir, "world_symbol_usage.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"signs": signs}, f, ensure_ascii=False, indent=2)

    print(f"[world_symbol_usage] Wrote {len(signs)} signs to: {out_path}")


if __name__ == "__main__":
    run_world_usage_demo()
