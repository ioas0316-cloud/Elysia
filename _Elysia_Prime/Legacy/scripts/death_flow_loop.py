# [Genesis: 2025-12-02] Purified by Elysia
#!/usr/bin/env python
"""Caretaker automation for Death Flow / Corpse-to-Memory world kit."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict


DEATH_EVENTS = [
    "corpse_returned",
    "ancestor_song",
    "caretaker_guidance",
    "memory_seed_planted",
    "river_of_light",
    "silent_watch",
]


def build_death_flow_snapshot(seed: int, rng: random.Random, minutes_per_tick: float) -> Dict[str, object]:
    release = round(rng.uniform(0.1, 0.9), 3)
    integration = round(rng.uniform(0.2, 0.95), 3)
    grief = round(rng.uniform(0.05, 0.8), 3)
    event = rng.choice(DEATH_EVENTS)
    return {
        "seed": seed,
        "world_kit": "DEATH_FLOW",
        "body_architecture": "memory_relay",
        "minutes_per_tick": minutes_per_tick,
        "release_index": release,
        "integration_index": integration,
        "grief_pressure": grief,
        "event": event,
    }


def run_death_flow_loop(seeds: int, minutes_per_tick: float, output: Path, sleep_seconds: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for seed in range(seeds):
            rng = random.Random(seed + 30_000)
            snapshot = build_death_flow_snapshot(seed, rng, minutes_per_tick)
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            print(
                f"[death_flow_loop] seed={seed} release={snapshot['release_index']} "
                f"integration={snapshot['integration_index']}"
            )
            time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Death Flow (corpse-to-memory) logging.")
    parser.add_argument("--seeds", type=int, default=20, help="number of seeds (default: 20)")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=30.0,
        help="minutes per tick reference (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/death_flow_loop.jsonl"),
        help="output JSONL file path",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="seconds to wait between seeds (default: 0.05)",
    )
    args = parser.parse_args()
    run_death_flow_loop(args.seeds, args.time_scale, args.output, args.sleep)


if __name__ == "__main__":
    main()