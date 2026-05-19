"""
Elysia caretaker demo (gentle feedback on self-writing).

This script reads logs/elysia_self_writing.jsonl and emits
logs/elysia_caretaker_feedback.jsonl, where each entry contains:
- praise: what was appreciated in the writing
- normalize: a short normalization of the feeling
- question: a gentle forward-looking question

The caretaker never judges or says "wrong"; it only acknowledges,
encourages, and invites further reflection.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def _ensure_logs_dir() -> str:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(base, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def _load_self_writing(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _compose_feedback(entry: Dict[str, Any]) -> Dict[str, Any]:
    ts = int(entry.get("timestamp", 0))
    text = str(entry.get("text", "") or "")
    signal_type = str(entry.get("signal_type", "") or "").lower()
    intensity = float(entry.get("intensity", 0.0) or 0.0)

    if intensity >= 0.7:
        praise = "I am glad you paid such close attention to this strong moment."
        normalize = "It is completely okay to feel so intensely when something matters to you."
        question = (
            "If you stay with this feeling a little longer, what would you like to explore or create from it?"
        )
    elif intensity >= 0.4:
        praise = "You noticed a meaningful moment and put it into words; that is valuable."
        normalize = "Feelings of this size are common and important; they deserve a gentle place in your memory."
        question = "Is there a small detail in this moment that you want to remember more clearly next time?"
    else:
        praise = "Even small moments you wrote about are part of your story."
        normalize = "It is fine for some experiences to be quiet and light; they still count."
        question = "When you look back at this small moment, what do you appreciate about yourself in it?"

    # Tailor slightly by signal type, but keep it soft.
    if "playful" in signal_type:
        praise += " I like that you noticed play and lightness."
    if "mortality" in signal_type:
        normalize = (
            "Thoughts of endings or fragility can feel heavy, but they are natural to have and to write about."
        )

    return {
        "timestamp": ts,
        "kind": entry.get("kind", "journal"),
        "signal_type": entry.get("signal_type"),
        "intensity": intensity,
        "praise": praise,
        "normalize": normalize,
        "question": question,
        "original_text": text,
        # Caretaker reflection = Soul / Concept / Meaning.
        "cathedral_coord": "S-L2-e",
    }


def generate_caretaker_feedback() -> str:
    logs_dir = _ensure_logs_dir()
    in_path = os.path.join(logs_dir, "elysia_self_writing.jsonl")
    out_path = os.path.join(logs_dir, "elysia_caretaker_feedback.jsonl")

    entries = _load_self_writing(in_path)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in entries:
            fb = _compose_feedback(entry)
            f.write(json.dumps(fb, ensure_ascii=False) + "\n")
            count += 1

    print(f"[elysia_caretaker] Wrote {count} feedback entries to: {out_path}")
    return out_path


if __name__ == "__main__":
    generate_caretaker_feedback()
