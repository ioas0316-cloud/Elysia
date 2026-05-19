#!/usr/bin/env python
"""Caretaker automation for Memory Circulation protocol."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict


MEMORY_CHANNELS = ["symbol_episode", "text_episode", "causal_episode", "persona_whisper"]


def build_memory_snapshot(seed: int, rng: random.Random, minutes_per_tick: float) -> Dict[str, object]:
    circulation = round(rng.uniform(0.2, 0.98), 3)
    residue = round(rng.uniform(0.05, 0.6), 3)
    channel = rng.choice(MEMORY_CHANNELS)
    diversity = round(rng.uniform(0.3, 0.95), 3)
    return {
        "seed": seed,
        "world_kit": "MEMORY_CIRCULATION",
        "body_architecture": "concept_kernel",
        "minutes_per_tick": minutes_per_tick,
        "circulation_strength": circulation,
        "residue_index": residue,
        "dominant_channel": channel,
        "diversity_score": diversity,
    }


def run_memory_circulation_loop(seeds: int, minutes_per_tick: float, output: Path, sleep_seconds: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for seed in range(seeds):
            rng = random.Random(seed + 40_000)
            snapshot = build_memory_snapshot(seed, rng, minutes_per_tick)
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            print(
                f"[memory_circulation_loop] seed={seed} channel={snapshot['dominant_channel']} "
                f"circulation={snapshot['circulation_strength']}"
            )
            time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Memory Circulation logging.")
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
        default=Path("logs/memory_circulation_loop.jsonl"),
        help="output JSONL file path",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="seconds to wait between seeds (default: 0.05)",
    )
    args = parser.parse_args()
    run_memory_circulation_loop(args.seeds, args.time_scale, args.output, args.sleep)


if __name__ == "__main__":
    main()
