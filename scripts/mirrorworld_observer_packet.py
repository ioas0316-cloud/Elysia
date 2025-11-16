"""
MirrorWorld observer packet builder.

This script packages Elysia's self-writing and caretaker feedback into
a compact JSON structure that a tutor/observer (e.g. transformer_tutor)
can ingest for scoring and guidance.

Output: logs/mirrorworld_observer_packet.json
  {
    "timestamp": "...",
    "self_writing": [...],
    "caretaker_feedback": [...],
    "language_field_snapshot": {...}
  }
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from scripts.elysia_language_field import load_language_field


def _ensure_logs_dir() -> str:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(base, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
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


def build_observer_packet() -> str:
    logs_dir = _ensure_logs_dir()
    self_path = os.path.join(logs_dir, "elysia_self_writing.jsonl")
    caretaker_path = os.path.join(logs_dir, "elysia_caretaker_feedback.jsonl")
    lang_field_path = os.path.join(logs_dir, "elysia_language_field.json")

    self_entries = _load_jsonl(self_path)
    caretaker_entries = _load_jsonl(caretaker_path)
    lang_field = load_language_field(lang_field_path)

    ts = datetime.now(timezone.utc).isoformat()
    packet = {
        "timestamp": ts,
        "self_writing": self_entries,
        "caretaker_feedback": caretaker_entries,
        "language_field_snapshot": lang_field.to_dict(),
    }

    out_path = os.path.join(logs_dir, "mirrorworld_observer_packet.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(packet, f, ensure_ascii=False, indent=2)

    print(f"[mirrorworld_observer_packet] Wrote observer packet to: {out_path}")
    return out_path


if __name__ == "__main__":
    build_observer_packet()

