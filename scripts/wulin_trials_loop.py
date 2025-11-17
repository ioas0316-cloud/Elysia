#!/usr/bin/env python
"""Caretaker automation for Wulin/Twin Villages trials."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict


WULIN_DUELS = [
    "sword_vs_spear",
    "inner_force_duel",
    "mountain_pass_standoff",
    "fair_trade_negotiation",
    "betrayal_unmasked",
    "alliance_ritual",
]

HONOR_SHIFTS = ["rise", "fall", "steady"]


def build_wulin_snapshot(seed: int, rng: random.Random, minutes_per_tick: float) -> Dict[str, object]:
    tension = round(rng.uniform(0.2, 0.95), 3)
    cooperation = round(rng.uniform(0.1, 0.85), 3)
    duel = rng.choice(WULIN_DUELS)
    honor_delta = rng.choice(HONOR_SHIFTS)
    return {
        "seed": seed,
        "world_kit": "WULINWORLD",
        "body_architecture": "martial_field",
        "minutes_per_tick": minutes_per_tick,
        "tension_index": tension,
        "cooperation_index": cooperation,
        "duel_event": duel,
        "honor_shift": honor_delta,
    }


def run_wulin_trials_loop(seeds: int, minutes_per_tick: float, output: Path, sleep_seconds: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for seed in range(seeds):
            rng = random.Random(seed + 20_000)
            snapshot = build_wulin_snapshot(seed, rng, minutes_per_tick)
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            print(
                f"[wulin_trials_loop] seed={seed} duel={snapshot['duel_event']} "
                f"tension={snapshot['tension_index']}"
            )
            time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Wulin/Twin Villages trials logging.")
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
        default=Path("logs/wulin_trials_loop.jsonl"),
        help="output JSONL file path",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="seconds to wait between seeds (default: 0.05)",
    )
    args = parser.parse_args()
    run_wulin_trials_loop(args.seeds, args.time_scale, args.output, args.sleep)


if __name__ == "__main__":
    main()
