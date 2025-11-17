#!/usr/bin/env python
"""Caretaker automation for FairyWorld ritual/mana tracking."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List


FAIRY_EVENTS = [
    "mana_bloom",
    "moonlit_ritual",
    "spirit_convergence",
    "grove_healing",
    "echo_harvest",
    "song_of_light",
]


def build_fairy_snapshot(seed: int, rng: random.Random, minutes_per_tick: float) -> Dict[str, object]:
    ritual_energy = round(rng.uniform(0.4, 0.98), 3)
    mana_flow = round(rng.uniform(0.2, 0.85), 3)
    resonance = round((ritual_energy + mana_flow) / 2.0, 3)
    events: List[str] = rng.sample(FAIRY_EVENTS, k=2)
    return {
        "seed": seed,
        "world_kit": "FAIRYWORLD",
        "body_architecture": "ritual_field",
        "minutes_per_tick": minutes_per_tick,
        "ritual_energy": ritual_energy,
        "mana_flow": mana_flow,
        "resonance_index": resonance,
        "notable_events": events,
    }


def run_fairyworld_growth_loop(seeds: int, minutes_per_tick: float, output: Path, sleep_seconds: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for seed in range(seeds):
            rng = random.Random(seed + 10_000)
            snapshot = build_fairy_snapshot(seed, rng, minutes_per_tick)
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            print(
                f"[fairyworld_growth_loop] seed={seed} ritual={snapshot['ritual_energy']} "
                f"mana={snapshot['mana_flow']}"
            )
            time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate FairyWorld ritual logging.")
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
        default=Path("logs/fairyworld_growth_loop.jsonl"),
        help="output JSONL file path",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="seconds to wait between seeds (default: 0.05)",
    )
    args = parser.parse_args()
    run_fairyworld_growth_loop(args.seeds, args.time_scale, args.output, args.sleep)


if __name__ == "__main__":
    main()
