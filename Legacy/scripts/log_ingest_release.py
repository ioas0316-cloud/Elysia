"""
Project Sophia release-log ingestion for CodeWorld / causal episodes.

This script is a bridge between high-level release telemetry and the
CodeWorld / CausalEpisode ecosystem used in Project Elysia.

It expects a JSONL input file where each line is a release log entry with
at least:
  {
    "ts": "2024-05-16T09:00:00Z",
    "component": "Project_Sophia.core.world",
    "version": "v1.2.3",
    "change_type": "bugfix|feature|refactor",
    "summary": "Short human-readable description",
    "details": { ... optional ... }
  }

The script converts these into a coarse causal episode stream that
CodeWorld / analysis scripts can consume:
  - logs/codeworld_causal_episodes.jsonl   (CodeEpisode-like dicts)
  - logs/causal_episodes_from_release.jsonl (aligned with CausalEpisode schema)

If the exact schema of the release logs differs, adjust the field mapping
in `_map_release_to_code_episode` and `_map_release_to_causal_episode`.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List

from scripts.causal_episodes import CausalEpisode


def _ensure_logs_dir() -> str:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(base, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def _load_release_log(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


@dataclass
class CodeEpisode:
    timestamp: int
    event_type: str
    entity_id: str
    kind: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": int(self.timestamp),
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "kind": self.kind,
            "data": dict(self.data or {}),
        }


def _parse_ts_to_int(ts: str | None) -> int:
    # Simple coarse mapping: drop to seconds since epoch-like counter.
    # For now, we just strip non-digits and keep last 10 characters as int if possible.
    if not ts:
        return 0
    digits = "".join(ch for ch in ts if ch.isdigit())
    if not digits:
        return 0
    try:
        return int(digits[-10:])
    except ValueError:
        return 0


def _map_release_to_code_episode(entry: Dict[str, Any]) -> CodeEpisode:
    ts = entry.get("ts") or entry.get("timestamp")
    timestamp = _parse_ts_to_int(str(ts))
    component = str(entry.get("component", "unknown") or "unknown")
    version = str(entry.get("version", "") or "")
    change_type = str(entry.get("change_type", "change") or "change").lower()
    summary = str(entry.get("summary", "") or "")
    details = entry.get("details", {}) or {}

    event_type = {
        "bugfix": "BUG_FIXED",
        "feature": "FEATURE_RELEASED",
        "refactor": "REFACTOR_APPLIED",
    }.get(change_type, "RELEASE_CHANGE")

    return CodeEpisode(
        timestamp=timestamp,
        event_type=event_type,
        entity_id=component,
        kind="module",
        data={
            "component": component,
            "version": version,
            "change_type": change_type,
            "summary": summary,
            "details": details,
        },
    )


def _map_release_to_causal_episode(entry: Dict[str, Any]) -> CausalEpisode:
    ts_raw = entry.get("ts") or entry.get("timestamp")
    timestamp = _parse_ts_to_int(str(ts_raw))
    component = str(entry.get("component", "unknown") or "unknown")
    change_type = str(entry.get("change_type", "change") or "change").lower()
    summary = str(entry.get("summary", "") or "")

    # We treat each release as a high-level "CAUSE" event affecting a "target" module.
    actor_id = "release_pipeline"
    target_id = component
    context = {
        "source": "release_log",
        "component": component,
        "change_type": change_type,
    }

    # Simple result_type mapping for now.
    if change_type == "bugfix":
        result_type = "improve_stability"
    elif change_type == "feature":
        result_type = "increase_complexity"
    elif change_type == "refactor":
        result_type = "reshape_structure"
    else:
        result_type = "release_change"

    actor_state = {
        "id": actor_id,
        "job_id": "pipeline.release",
        "power_score": 0.0,
    }
    target_state = {
        "id": target_id,
        "job_id": "module",
        "power_score": 0.0,
    }

    return CausalEpisode(
        timestamp=timestamp,
        event_type="RELEASE_CAUSAL",
        actor_id=actor_id,
        target_id=target_id,
        actor_state=actor_state,
        target_state=target_state,
        context={**context, "summary": summary},
        result_type=result_type,
    )


def ingest_release_log(
    input_path: str = "logs/project_sophia_release_log.jsonl",
) -> Dict[str, str]:
    """
    Ingest a Project Sophia release-log JSONL file and emit:
      - logs/codeworld_causal_episodes.jsonl
      - logs/causal_episodes_from_release.jsonl
    """
    logs_dir = _ensure_logs_dir()
    codeworld_path = os.path.join(logs_dir, "codeworld_causal_episodes.jsonl")
    causal_path = os.path.join(logs_dir, "causal_episodes_from_release.jsonl")

    entries = list(_load_release_log(input_path))

    with open(codeworld_path, "w", encoding="utf-8") as f_code, open(
        causal_path, "w", encoding="utf-8"
    ) as f_causal:
        for entry in entries:
            code_ep = _map_release_to_code_episode(entry)
            f_code.write(json.dumps(code_ep.to_dict(), ensure_ascii=False) + "\n")

            causal_ep = _map_release_to_causal_episode(entry)
            f_causal.write(json.dumps(asdict(causal_ep), ensure_ascii=False) + "\n")

    print(f"[log_ingest_release] Wrote {len(entries)} entries to:")
    print(f"  - {codeworld_path}")
    print(f"  - {causal_path}")
    return {"codeworld": codeworld_path, "causal": causal_path}


if __name__ == "__main__":
    ingest_release_log()

