# [Genesis: 2025-12-02] Purified by Elysia
#!/usr/bin/env python
"""Caretaker automation for CELLWORLD growth runs."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List


CELLWORLD_EVENTS = [
    "settlement_expansion",
    "harvest_festival",
    "storm_cycle",
    "civic_ceremony",
    "new_trade_route",
    "quiet_meditation",
]


def build_cellworld_snapshot(seed: int, rng: random.Random, minutes_per_tick: float) -> Dict[str, object]:
    living = rng.randint(80, 220)
    archived = rng.randint(10, 80)
    value_mass = round(rng.uniform(0.3, 0.95), 3)
    will_field = round(rng.uniform(0.25, 0.9), 3)
    events: List[str] = rng.sample(CELLWORLD_EVENTS, k=2)
    return {
        "seed": seed,
        "world_kit": "CELLWORLD",
        "body_architecture": "flow_field",
        "minutes_per_tick": minutes_per_tick,
        "living_cells": living,
        "archived_cells": archived,
        "value_mass_index": value_mass,
        "will_field_index": will_field,
        "notable_events": events,
    }


def run_cellworld_growth_loop(seeds: int, minutes_per_tick: float, output: Path, sleep_seconds: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for seed in range(seeds):
            rng = random.Random(seed)
            snapshot = build_cellworld_snapshot(seed, rng, minutes_per_tick)
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            print(
                f"[cellworld_growth_loop] seed={seed} living={snapshot['living_cells']} "
                f"value_mass={snapshot['value_mass_index']}"
            )
            time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate CELLWORLD growth logging.")
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
        default=Path("logs/cellworld_growth_loop.jsonl"),
        help="output JSONL file path",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="seconds to wait between seeds (default: 0.05)",
    )
    args = parser.parse_args()
    run_cellworld_growth_loop(args.seeds, args.time_scale, args.output, args.sleep)


if __name__ == "__main__":
    main()