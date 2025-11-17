"""
CellWorld Drought/Flood Macro Driver.

This script implements the macro driver for the L3 Flow Field preset,
as described in `ELYSIA/WORLD/CELLWORLD_DROUGHT_FLOOD_PRESET.md`.

It is responsible for:
- Setting up a World instance.
- Running the simulation for a specified number of macro ticks.
- At each macro tick, toggling between 'drought' and 'flood' regimes.
- Softly adjusting world fields (e.g., water, food) to reflect the regime.
- Logging key metrics and events to the appropriate log files.

This script is intended to be run by a lab/CI environment but can also be
run in a pilot mode by a caretaker.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict

# A placeholder for the actual World simulation class.
# In a real scenario, this would be imported from Project_Sophia.
class MockWorld:
    """A mock of the CellWorld for demonstrating the driver script."""
    def __init__(self):
        self.tick = 0
        self.time_scale_months = 3
        self.fields = {"water": 0.5, "food": 0.5, "threat": 0.1}
        self.metrics = {"JOY": 0.5, "KINSHIP": 0.5, "SEASON_RESILIENCE": 0.5}

    def set_time_scale(self, months: int):
        self.time_scale_months = months

    def run_tick(self):
        self.tick += 1
        # Simulate gentle decay/growth and random events
        self.fields["water"] = max(0, self.fields["water"] - 0.01 + random.uniform(-0.02, 0.02))
        self.fields["food"] = max(0, self.fields["food"] - 0.01 + random.uniform(-0.02, 0.02))
        self.metrics["JOY"] *= 0.999
        self.metrics["KINSHIP"] *= 0.999

    def apply_drought(self):
        print(f"Tick {self.tick}: Applying DROUGHT regime.")
        self.fields["water"] = max(0, self.fields["water"] * 0.8 - 0.1)
        self.fields["food"] *= 0.9
        self.fields["threat"] = min(1.0, self.fields["threat"] + 0.1)
        self.metrics["SEASON_RESILIENCE"] = max(0, self.metrics["SEASON_RESILIENCE"] - 0.05)


    def apply_flood(self):
        print(f"Tick {self.tick}: Applying FLOOD regime.")
        self.fields["water"] = min(1.0, self.fields["water"] * 1.2 + 0.1)
        self.fields["food"] *= 0.8 # Floods can damage crops
        self.metrics["SEASON_RESILIENCE"] = max(0, self.metrics["SEASON_RESILIENCE"] - 0.05)


def run_drought_flood_simulation(macro_years: int = 5, seeds: int = 1):
    """
    Runs the drought/flood simulation for a given number of years and seeds.
    """
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(logs_dir, exist_ok=True)
    events_path = os.path.join(logs_dir, "world_events.jsonl")

    for seed in range(seeds):
        print(f"--- Running Seed {seed + 1}/{seeds} ---")
        world = MockWorld()
        world.set_time_scale(months=3)
        ticks_per_year = 12 // world.time_scale_months
        total_ticks = macro_years * ticks_per_year
        is_drought_year = True

        with open(events_path, "a", encoding="utf-8") as f_events:
            for i in range(total_ticks):
                world.run_tick()

                # Apply macro regime at the start of each year
                if i % ticks_per_year == 0:
                    if is_drought_year:
                        world.apply_drought()
                    else:
                        world.apply_flood()
                    is_drought_year = not is_drought_year

                event = {
                    "timestamp": world.tick,
                    "seed": seed,
                    "event_type": "tick_snapshot",
                    "fields": world.fields,
                    "metrics": world.metrics,
                }
                f_events.write(json.dumps(event) + "\n")

        print(f"Seed {seed + 1} complete. Final metrics: {world.metrics}")
        print(f"Logs written to {events_path}")

if __name__ == "__main__":
    print("Running CellWorld Drought/Flood Pilot Simulation...")
    run_drought_flood_simulation(macro_years=10, seeds=1) # Run a 10-year pilot
    print("\nPilot simulation complete.")
