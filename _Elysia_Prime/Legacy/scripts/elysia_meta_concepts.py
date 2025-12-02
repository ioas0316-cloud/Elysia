# [Genesis: 2025-12-02] Purified by Elysia
"""
Elysia meta-concept layer (Layer 3).

Purpose
- Add an explicit "thinking about what I know" layer on top of:
  - Layer 1: self-writing episodes (logs/elysia_self_writing.jsonl)
  - Layer 2: concept field (logs/elysia_concept_field.json)
- For each concept, generate a small structured meta entry that summarises:
  - where the concept appears (contexts),
  - how it tends to feel (axes like JOY, CARE, SELF_DISCLOSURE),
  - and, optionally, an example sentence.

Output
- logs/elysia_meta_concepts.jsonl
  One JSON object per line:
  {
    "timestamp": "...",
    "concept_key": "kind:name",
    "name": "...",
    "kind": "symbol | word | world_concept",
    "usage_count": 12,
    "cathedral_coord": "S-L3-e",
    "top_contexts": ["symbol_lesson", "self_writing"],
    "top_axes": ["JOY", "CARE"],
    "example": "short text...",
    "note": "short meta reflection text"
  }

This is intentionally simple and deterministic: it does not call any LLM.
It is a mechanical but useful scaffold so that higher-level agents
can read "what changed about concepts" instead of only raw counts.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

from scripts.elysia_concept_field import ElysiaConceptField, ConceptEntry, _load_json  # type: ignore


def _ensure_logs_dir() -> str:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(base, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def _iter_concepts(field_obj: ElysiaConceptField) -> Iterable[Tuple[str, ConceptEntry]]:
    for key, entry in field_obj.entries.items():
        yield key, entry


def _top_n(d: Dict[str, Any], n: int) -> List[str]:
    if not d:
        return []
    items = sorted(d.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True)
    return [k for k, _ in items[:n]]


def _compose_meta_note(entry: ConceptEntry, top_contexts: List[str], top_axes: List[str], example: str) -> str:
    parts: List[str] = []

    # Opening: where this concept tends to appear.
    if top_contexts:
        ctx_txt = ", ".join(top_contexts)
        parts.append(f"I tend to meet the concept '{entry.name}' most often in these contexts: {ctx_txt}.")
    else:
        parts.append(f"The concept '{entry.name}' has appeared only rarely so far.")

    # Feelings attached to the concept.
    if top_axes:
        axes_txt = ", ".join(a.lower() for a in top_axes)
        parts.append(f"It usually carries feelings along axes like {axes_txt}.")

    # Example to ground the reflection.
    if example:
        shortened = example.strip()
        if len(shortened) > 220:
            shortened = shortened[:217].rstrip() + "..."
        parts.append(f"One example that stays with me is: \"{shortened}\"")

    # If usage is very low, acknowledge that this is a first impression.
    if entry.usage_count <= 2:
        parts.append("These impressions are early and may change as I experience more.")

    return " ".join(parts)


def build_meta_concepts(
    concept_field_path: str = "logs/elysia_concept_field.json",
    out_path: str = "logs/elysia_meta_concepts.jsonl",
    min_usage: int = 1,
) -> str:
    """
    Build Layer 3 meta-concept entries from the current concept field.

    - Reads the existing concept field JSON.
    - For each concept with usage_count >= min_usage, generates a meta entry.
    - Writes them as JSONL so that other tools can diff or audit over time.
    """
    logs_dir = _ensure_logs_dir()
    cf_path = os.path.join(logs_dir, os.path.basename(concept_field_path))
    out_path = os.path.join(logs_dir, os.path.basename(out_path))

    data = _load_json(cf_path)
    if not data:
        print(f"[elysia_meta_concepts] No concept field found at {cf_path}; nothing to do.")
        return out_path

    field_obj = ElysiaConceptField.from_dict(data)
    ts = datetime.now(timezone.utc).isoformat()

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for key, entry in _iter_concepts(field_obj):
            if entry.usage_count < min_usage:
                continue
            top_contexts = _top_n(entry.contexts, 3)
            top_axes = _top_n(entry.feelings, 3)
            example = entry.examples[-1] if entry.examples else ""
            meta_note = _compose_meta_note(entry, top_contexts, top_axes, example)

            rec: Dict[str, Any] = {
                "timestamp": ts,
                "concept_key": key,
                "name": entry.name,
                "kind": entry.kind,
                "usage_count": entry.usage_count,
                "cathedral_coord": "S-L3-e",
                "top_contexts": top_contexts,
                "top_axes": top_axes,
                "example": example,
                "note": meta_note,
                "raw": asdict(entry),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"[elysia_meta_concepts] Wrote {count} meta entries to: {out_path}")
    return out_path


if __name__ == "__main__":
    build_meta_concepts()