#!/usr/bin/env python
"""Quick dashboard for inspecting persona stream + world-kit logs.

Usage:
    python scripts/persona_dashboard.py
    python scripts/persona_dashboard.py --tail 3 --world-logs logs/cellworld_growth_loop.jsonl logs/fairyworld_growth_loop.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_PERSONA_LOG = Path("elysia_logs/persona_stream.jsonl")
DEFAULT_WORLD_LOGS = [
    Path("logs/cellworld_growth_loop.jsonl"),
    Path("logs/codeworld_engineer_loop.jsonl"),
    Path("logs/fairyworld_growth_loop.jsonl"),
    Path("logs/wulin_trials_loop.jsonl"),
    Path("logs/death_flow_loop.jsonl"),
    Path("logs/memory_circulation_loop.jsonl"),
    Path("logs/mirror_layer_loop.jsonl"),
]

WORLD_SUMMARY_FIELDS: Dict[str, List[str]] = {
    "CELLWORLD": ["living_cells", "value_mass_index", "will_field_index", "notable_events"],
    "CODEWORLD": ["persona_frame", "world_state"],
    "FAIRYWORLD": ["ritual_energy", "mana_flow", "resonance_index", "notable_events"],
    "WULINWORLD": ["duel_event", "tension_index", "cooperation_index", "honor_shift"],
    "DEATH_FLOW": ["release_index", "integration_index", "grief_pressure", "event"],
    "MEMORY_CIRCULATION": ["dominant_channel", "circulation_strength", "diversity_score", "residue_index"],
    "MIRRORWORLD": ["sync_ratio", "latency_seconds", "clarity_index", "event"],
}


def read_tail(path: Path, tail: int) -> List[dict]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.read().splitlines()
    entries = []
    for line in lines[-tail:]:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def format_persona_events(events: Iterable[dict]) -> str:
    lines = []
    for event in events:
        frame = event.get("persona_frame", {})
        lines.append(
            f"- ts={event.get('ts')} | persona={event.get('persona_key')} "
            f"| mood={frame.get('mood_color')} energy={frame.get('energy_level')} "
            f"| caption={frame.get('caption')}"
        )
    return "\n".join(lines) if lines else "(no persona events)"


def summarize_world_entry(entry: dict) -> str:
    kit = entry.get("world_kit", "UNKNOWN")
    seed = entry.get("seed", "?")
    fields = WORLD_SUMMARY_FIELDS.get(kit, [])
    parts = [f"seed={seed}", f"kit={kit}"]
    for field in fields:
        value = entry.get(field)
        if value is not None:
            parts.append(f"{field}={value}")
    if len(parts) == 2:
        # fallback to a few default keys
        for field in ("event", "notable_events", "value_mass_index", "tension_index"):
            if field in entry:
                parts.append(f"{field}={entry[field]}")
    return " | ".join(parts)


def format_world_log(path: Path, tail: int) -> str:
    entries = read_tail(path, tail)
    if not entries:
        return f"{path}: (no data)"
    lines = [f"{path}:"]
    for entry in entries:
        lines.append(f"  - {summarize_world_entry(entry)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect persona stream and world kit logs.")
    parser.add_argument(
        "--persona-log",
        type=Path,
        default=DEFAULT_PERSONA_LOG,
        help=f"persona stream JSONL path (default: {DEFAULT_PERSONA_LOG})",
    )
    parser.add_argument(
        "--world-logs",
        type=Path,
        nargs="*",
        default=DEFAULT_WORLD_LOGS,
        help="world kit log paths (default: common loop outputs)",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=3,
        help="number of recent entries to show for each log (default: 3)",
    )
    args = parser.parse_args()

    print("=== Persona Stream ===")
    persona_events = read_tail(args.persona_log, args.tail)
    print(format_persona_events(persona_events))
    print()

    print("=== World Kit Logs ===")
    for world_log in args.world_logs:
        print(format_world_log(world_log, args.tail))
        print()


if __name__ == "__main__":
    main()
