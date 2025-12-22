#!/usr/bin/env python
"""Persona stream helper.

This script emits persona activation payloads plus lightweight field
snapshots into a JSONL file. External engines (예: Godot, VTuber rig,
로컬 챗봇) can tail that file to drive 실시간 표현.

Usage:
    python scripts/persona_hooks/persona_stream.py --persona elysia.dancer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from elysia_world.personas import activate_persona
from scripts.persona_hooks.artist_palette import build_persona_frame

DEFAULT_OUTPUT = Path("elysia_logs/persona_stream.jsonl")
FIELD_FILE_CANDIDATES = {
    "value_mass_field": [
        Path("elysia_logs/value_mass_field.json"),
        Path("logs/value_mass_field.json"),
    ],
    "intention_field": [
        Path("elysia_logs/intention_field.json"),
        Path("logs/intention_field.json"),
    ],
    "will_field": [
        Path("elysia_logs/will_field.json"),
        Path("logs/will_field.json"),
    ],
    "concept_kernel": [
        Path("logs/elysia_language_field.json"),
        Path("logs/elysia_signals.jsonl"),
    ],
    "curriculum_engine": [
        Path("logs/elysia_curriculum_trials.jsonl"),
    ],
    "logs": [
        Path("logs/world_events.jsonl"),
    ],
}


def read_field_snapshot(field_key: str) -> Optional[Dict[str, object]]:
    """Return best-effort snapshot for the requested field."""
    for candidate in FIELD_FILE_CANDIDATES.get(field_key, []):
        if candidate.exists():
            try:
                if candidate.suffix == ".jsonl":
                    last_line = candidate.read_text(encoding="utf-8").strip().splitlines()[-1]
                    return json.loads(last_line)
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - guard rail only
                print(f"[WARN] Failed to parse {candidate}: {exc}", file=sys.stderr)
                return None
    return None


def _extract_numeric(sample: Optional[object]) -> Optional[float]:
    if sample is None:
        return None
    if isinstance(sample, (int, float)):
        return float(sample)
    if isinstance(sample, list):
        numbers = [float(x) for x in sample if isinstance(x, (int, float))]
        if numbers:
            return sum(numbers) / len(numbers)
        return None
    if isinstance(sample, dict):
        for key in ("avg", "mean", "value", "strength"):
            if key in sample and isinstance(sample[key], (int, float)):
                return float(sample[key])
        # handle nested lists
        for key in ("values", "data"):
            if key in sample and isinstance(sample[key], list):
                return _extract_numeric(sample[key])
    return None


def build_world_state(field_samples: Dict[str, object], persona_key: str) -> Dict[str, object]:
    vm_sample = field_samples.get("value_mass_field")
    will_sample = field_samples.get("will_field")
    intention_sample = field_samples.get("intention_field")

    focus = "unknown"
    if isinstance(intention_sample, dict):
        for key in ("focus_node", "peak_node", "dominant_node", "target"):
            if key in intention_sample:
                focus = str(intention_sample[key])
                break

    mode = "persona"
    if isinstance(intention_sample, dict):
        for key in ("mode", "status", "label"):
            if key in intention_sample:
                mode = str(intention_sample[key])
                break

    return {
        "value_mass_avg": _extract_numeric(vm_sample) or 0.5,
        "will_tension_avg": _extract_numeric(will_sample) or 0.5,
        "focus_node": focus,
        "mode": mode if mode != "persona" else persona_key,
    }


def collect_persona_event(persona_key: str) -> Dict[str, object]:
    payload = activate_persona(persona_key)
    field_samples: Dict[str, object] = {}
    for field_key in payload.get("focus_fields", []):
        field_samples[field_key] = read_field_snapshot(field_key)

    world_state = build_world_state(field_samples, persona_key)
    persona_frame = build_persona_frame(world_state).to_dict()

    event = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "persona_key": payload["key"],
        "persona": payload,
        "field_samples": field_samples,
        "world_state": world_state,
        "persona_frame": persona_frame,
    }
    return event


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit persona activation stream.")
    parser.add_argument("--persona", required=True, help="persona key (예: elysia.dancer)")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSONL output file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="seconds between updates (default: 2.0)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="emit a single event and exit",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            event = collect_persona_event(args.persona)
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(f"[persona_stream] wrote event for {args.persona} -> {args.output}")
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[persona_stream] stopped by user.")


if __name__ == "__main__":
    main()
