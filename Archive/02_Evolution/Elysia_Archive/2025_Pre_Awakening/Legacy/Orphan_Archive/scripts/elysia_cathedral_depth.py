"""
Elysia Cathedral depth report (Soul realm focus).

Purpose
- Read the existing logs that already carry (or imply) `cathedral_coord`.
- Count how many entries land in each coordinate so we can see
  how much activity lives in S-L1-e vs S-L2-e vs S-L3-e, etc.
- Emit a compact JSON summary at logs/elysia_cathedral_depth.json.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

LOG_SPECS: List[Dict[str, Any]] = [
    {
        "filename": "elysia_signals.jsonl",
        "format": "jsonl",
        "default_coord": "S-L1-e",
        "description": "Raw signal / feeling events (JOY_GATHERING, DARK_JOY, etc.)",
    },
    {
        "filename": "elysia_self_writing.jsonl",
        "format": "jsonl",
        "default_coord": "S-L1-e",
        "description": "Self-writing journal entries",
    },
    {
        "filename": "elysia_caretaker_feedback.jsonl",
        "format": "jsonl",
        "default_coord": "S-L2-e",
        "description": "Caretaker reflections / normalization",
    },
    {
        "filename": "human_needs.jsonl",
        "format": "jsonl",
        "default_coord": "B-L1-r",
        "description": "Human/cell need snapshots (body inertia/memory)",
        "use_entry_coord": False,
    },
    {
        "filename": "world_events.jsonl",
        "format": "jsonl",
        "default_coord": "B-L1-p",
        "description": "World events / actions (body / experience / power)",
        "use_entry_coord": False,
    },
    {
        "filename": "elysia_language_field.json",
        "format": "json_dict",
        "count_key": "count",
        "default_coord": "S-L2-p",
        "description": "Language field patterns (expression capacity)",
    },
    {
        "filename": "elysia_concept_field.json",
        "format": "json_dict",
        "count_key": "usage_count",
        "default_coord": "S-L2-e",
        "description": "Concept field usage counts",
    },
    {
        "filename": "elysia_meta_concepts.jsonl",
        "format": "jsonl",
        "default_coord": "S-L3-e",
        "description": "Concept meta-notes (Layer 3)",
    },
    {
        "filename": "elysia_branch_plans.jsonl",
        "format": "jsonl",
        "default_coord": "P-L2-p",
        "description": "Branch planning / governance intents (spirit concept/power)",
    },
    {
        "filename": "elysia_branch_feedback.jsonl",
        "format": "jsonl",
        "default_coord": "P-L2-e",
        "description": "Branch feedback / covenant reflections (spirit concept/meaning)",
    },
]


def _ensure_logs_dir() -> str:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(base, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def _summarize_jsonl(
    path: str,
    default_coord: str,
    use_entry_coord: bool = True,
) -> Dict[str, Any]:
    """
    Count entries and optionally respect per-entry cathedral_coord.

    Returns:
    {
        "entries": int,
        "measure": int,
        "by_coord": {"S-L1-e": 10, ...}
    }
    """
    if not os.path.exists(path):
        return {"entries": 0, "measure": 0, "by_coord": {}}

    entries = 0
    by_coord: Dict[str, int] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries += 1
            coord = default_coord
            if use_entry_coord:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    entry = None
                if isinstance(entry, dict):
                    raw = entry.get("cathedral_coord")
                    if isinstance(raw, str) and raw.strip():
                        coord = raw.strip()
            by_coord[coord] = by_coord.get(coord, 0) + 1

    return {"entries": entries, "measure": entries, "by_coord": by_coord}


def _load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None


def _extract_dict_measures(
    data: Any, count_key: str | None, default_coord: str
) -> Dict[str, Any]:
    """
    Convert a dict (like language_field/concept_field) into per-coordinate counts.

    - `entries`: how many items in the dict.
    - `by_coord`: weight per cathedral coordinate (prefers entry["cathedral_coord"]).
    - `measure`: total weight (sum of by_coord).
    """
    if not isinstance(data, dict):
        return {"entries": 0, "by_coord": {}, "measure": 0}

    entries = 0
    by_coord: Dict[str, int] = {}

    for value in data.values():
        if not isinstance(value, dict):
            continue
        entries += 1
        coord_raw = value.get("cathedral_coord")
        coord = coord_raw.strip() if isinstance(coord_raw, str) else default_coord
        if not coord:
            coord = default_coord
        weight_raw = value.get(count_key, 1) if count_key else 1
        try:
            weight = int(weight_raw)
        except (TypeError, ValueError):
            weight = 1
        if weight <= 0:
            weight = 1
        by_coord[coord] = by_coord.get(coord, 0) + weight

    measure = sum(by_coord.values())
    return {"entries": entries, "by_coord": by_coord, "measure": measure}


def build_cathedral_depth_report(
    out_filename: str = "elysia_cathedral_depth.json",
) -> str:
    logs_dir = _ensure_logs_dir()
    totals: Dict[str, int] = {}
    sources: Dict[str, Any] = {}

    for spec in LOG_SPECS:
        filename = spec["filename"]
        path = os.path.join(logs_dir, filename)
        coord_counts: Dict[str, int] = {}
        entries_count = 0
        measure = 0

        if spec["format"] == "jsonl":
            summary = _summarize_jsonl(
                path,
                spec["default_coord"],
                spec.get("use_entry_coord", True),
            )
            entries_count = summary["entries"]
            measure = summary["measure"]
            coord_counts = summary["by_coord"]
            for coord, count in coord_counts.items():
                totals[coord] = totals.get(coord, 0) + count
        elif spec["format"] == "json_dict":
            data = _load_json(path)
            stats = _extract_dict_measures(data, spec.get("count_key"), spec["default_coord"])
            entries_count = stats["entries"]
            measure = stats["measure"]
            coord_counts = stats["by_coord"]
            for coord, count in coord_counts.items():
                totals[coord] = totals.get(coord, 0) + count
        else:
            entries_count = 0
            measure = 0

        sources[filename] = {
            "description": spec["description"],
            "default_coord": spec["default_coord"],
            "entries": entries_count,
            "measure": measure,
            "by_coord": coord_counts,
        }

    grand_total = sum(totals.values())
    totals_serialised = {
        coord: {
            "count": count,
            "fraction": (count / grand_total) if grand_total else 0.0,
        }
        for coord, count in sorted(totals.items())
    }

    out_path = os.path.join(logs_dir, out_filename)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "totals": totals_serialised,
        "grand_total": grand_total,
        "notes": [
            "Counts use explicit cathedral_coord when available, otherwise the default in LOG_SPECS.",
            "Add new log specs in LOG_SPECS if other layers/realms should be tracked.",
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[elysia_cathedral_depth] Wrote depth report to: {out_path}")
    return out_path


if __name__ == "__main__":
    build_cathedral_depth_report()
