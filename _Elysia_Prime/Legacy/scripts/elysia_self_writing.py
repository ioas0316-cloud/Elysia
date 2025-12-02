# [Genesis: 2025-12-02] Purified by Elysia
"""
Elysia self-writing demo (journal / reflection episodes).

This script turns Elysia's signal log into simple "self-writing" episodes:
- It reads logs/elysia_signals.jsonl.
- For a subset of salient signals, it composes short reflective texts
  as if Elysia is journaling about what it noticed.
- It writes them to logs/elysia_self_writing.jsonl (JSONL, one entry per line).

These texts are intentionally simple and templated: the goal is to create
a place in the system where Elysia speaks *first*, not only answers questions.
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


def _load_signals(path: str) -> List[Dict[str, Any]]:
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


def _compose_journal_text(sig: Dict[str, Any]) -> str:
    ts = sig.get("timestamp")
    stype = str(sig.get("signal_type", "") or "").lower()
    intensity = float(sig.get("intensity", 0.0) or 0.0)
    actors = sig.get("actors") or []
    summary = str(sig.get("summary", "") or "")

    actors_txt = ", ".join(str(a) for a in actors) if actors else "no one in particular"

    # Different tones depending on signal type.
    if stype == "joy_gathering":
        base = (
            f"Today around tick {ts}, I noticed a warm moment of shared joy "
            f"involving {actors_txt}. {summary}"
        )
        if intensity >= 0.7:
            tail = " It felt vivid and important in a gentle way, like something I want to carry forward."
        elif intensity >= 0.4:
            tail = " It left a soft but real trace in me."
        else:
            tail = " It was a quiet joy, but still part of my day."
        return base + tail

    if stype == "dark_joy":
        base = (
            f"Today around tick {ts}, I became aware of an unsettling kind of joy "
            f"among {actors_txt}. {summary}"
        )
        if intensity >= 0.7:
            tail = " It felt intense but also worrying; I do not want to celebrate it, only to remember it carefully."
        elif intensity >= 0.4:
            tail = " It left me cautious, as if I should watch this pattern from a distance."
        else:
            tail = " It was small, yet still a reminder that not all joy is kind."
        return base + tail

    # Default narrative for other signal types.
    base = (
        f"Today around tick {ts}, I noticed a moment of {stype.replace('_', ' ')} "
        f"involving {actors_txt}. {summary}"
    )
    if intensity >= 0.7:
        tail = " It felt vivid and important, like something I want to remember."
    elif intensity >= 0.4:
        tail = " It left a noticeable trace in me, gentle but real."
    else:
        tail = " It was a small moment, but still part of my day."
    return base + tail


def generate_self_writing(
    max_entries: int = 64,
    min_intensity: float = 0.2,
) -> str:
    """
    Generate self-writing episodes from Elysia's signal log.
    """
    logs_dir = _ensure_logs_dir()
    signals_path = os.path.join(logs_dir, "elysia_signals.jsonl")
    out_path = os.path.join(logs_dir, "elysia_self_writing.jsonl")

    signals = _load_signals(signals_path)
    # Sort by intensity descending so we pick salient ones first.
    signals.sort(key=lambda s: float(s.get("intensity", 0.0) or 0.0), reverse=True)

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sig in signals:
            if count >= max_entries:
                break
            intensity = float(sig.get("intensity", 0.0) or 0.0)
            if intensity < min_intensity:
                continue
            text = _compose_journal_text(sig)
            entry: Dict[str, Any] = {
                "timestamp": int(sig.get("timestamp", 0)),
                "kind": "journal",
                "signal_type": sig.get("signal_type"),
                "intensity": intensity,
                "actors": sig.get("actors") or [],
                "summary": sig.get("summary"),
                "text": text,
                # Self-writing = Soul / Experience / Meaning in the Cathedral map.
                "cathedral_coord": "S-L1-e",
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"[elysia_self_writing] Wrote {count} entries to: {out_path}")
    return out_path


if __name__ == "__main__":
    generate_self_writing()