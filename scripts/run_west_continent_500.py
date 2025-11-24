"""
Spawn 500 agents with West Continent theme and run a short simulation.
Designed to be lightweight for 1060-class GPUs: small map, low-res bands, minimal logging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.kg_manager import KGManager
from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.world_themes.west_continent.config import WEST_THEME


def make_world(map_size: int = 128) -> World:
    kg = KGManager()
    wave = WaveMechanics(kg)
    world = World(primordial_dna={}, wave_mechanics=wave, logger=logging.getLogger("WestSim"))
    world.width = map_size
    # Tune for perf
    world.band_split_enabled = True
    world.band_low_resolution = 64
    world.micro_layer_enabled = False
    world.spectrum_log_interval = 100
    world.free_will_threat_threshold = 120.0
    world.free_will_value_threshold = 200.0
    world.trust_scarcity = 0.4
    world.trust_hunger_drain = 0.2
    return world


def spawn_agents(world: World, n: int = 500):
    jobs = WEST_THEME.preferred_job_ids
    labels = ["knight", "mage", "priest", "merchant", "blacksmith"]
    for i in range(n):
        job = jobs[i % len(jobs)]
        label = labels[i % len(labels)]
        cell_id = f"west_agent_{i:03d}"
        world.add_cell(
            concept_id=cell_id,
            properties={
                "label": label,
                "culture": "west",
                "vitality": 50,
                "wisdom": 30,
                "prestige": 5,
                "hp": 80,
                "max_hp": 80,
                "hydration": 80.0,
                "hunger": 80.0,
                "position": {"x": (i * 3) % world.width, "y": (i * 7) % world.width, "z": 0},
            },
        )


def run_sim(cycles: int = 300, map_size: int = 128):
    logging.basicConfig(level=logging.INFO)
    world = make_world(map_size)
    spawn_agents(world, 500)
    logger = world.logger
    logger.info(f"Spawned 500 agents on {map_size}x{map_size} west continent theme.")

    for t in range(cycles):
        world.run_simulation_step()
        if t % 50 == 0:
            alive = world.is_alive_mask.sum()
            avg_hp = world.hp[world.is_alive_mask].mean() if alive > 0 else 0.0
            logger.info(f"Tick {t}/{cycles}: alive={alive}, avg_hp={avg_hp:.1f}")

    logger.info("Simulation complete.")
    return world


if __name__ == "__main__":
    run_sim()
