#!/usr/bin/env python
"""Caretaker helper for CODEWORLD / engineer persona logging."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.persona_hooks.persona_stream import collect_persona_event  # type: ignore


def run_engineer_loop(seeds: int, minutes_per_tick: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for seed in range(seeds):
            event = collect_persona_event("elysia.engineer")
            payload = {
                "seed": seed,
                "minutes_per_tick": minutes_per_tick,
                "ts": event["ts"],
                "persona_frame": event["persona_frame"],
                "world_state": event["world_state"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(
                f"[elysia_engineer_loop] seed={seed} mood={payload['persona_frame']['mood_color']} "
                f"energy={payload['persona_frame']['energy_level']:.2f}"
            )
            time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CODEWORLD engineer persona logging loop.")
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds (default: 5)")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=30.0,
        help="minutes per tick for log reference (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/codeworld_engineer_loop.jsonl"),
        help="output JSONL file path",
    )
    args = parser.parse_args()
    run_engineer_loop(args.seeds, args.time_scale, args.output)


if __name__ == "__main__":
    main()
