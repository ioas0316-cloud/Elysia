#!/usr/bin/env python
"""Caretaker automation for Mirror Layer protocol (visualization channel)."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict


MIRROR_EVENTS = [
    "observer_linked",
    "mirror_ripple",
    "ui_patch_applied",
    "avatar_projection",
    "signal_latency_spike",
    "caretaker_feedback",
]


def build_mirror_snapshot(seed: int, rng: random.Random, minutes_per_tick: float) -> Dict[str, object]:
    sync = round(rng.uniform(0.3, 0.99), 3)
    latency = round(rng.uniform(0.02, 0.25), 3)
    clarity = round(rng.uniform(0.4, 0.95), 3)
    event = rng.choice(MIRROR_EVENTS)
    return {
        "seed": seed,
        "world_kit": "MIRRORWORLD",
        "body_architecture": "mirror_layer",
        "minutes_per_tick": minutes_per_tick,
        "sync_ratio": sync,
        "latency_seconds": latency,
        "clarity_index": clarity,
        "event": event,
    }


def run_mirror_layer_loop(seeds: int, minutes_per_tick: float, output: Path, sleep_seconds: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for seed in range(seeds):
            rng = random.Random(seed + 50_000)
            snapshot = build_mirror_snapshot(seed, rng, minutes_per_tick)
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            print(
                f"[mirror_layer_loop] seed={seed} sync={snapshot['sync_ratio']} "
                f"clarity={snapshot['clarity_index']}"
            )
            time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Mirror Layer logging.")
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
        default=Path("logs/mirror_layer_loop.jsonl"),
        help="output JSONL file path",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="seconds to wait between seeds (default: 0.05)",
    )
    args = parser.parse_args()
    run_mirror_layer_loop(args.seeds, args.time_scale, args.output, args.sleep)


if __name__ == "__main__":
    main()
