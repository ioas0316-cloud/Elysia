"""
West Continent simulation with configurable agent count/map size.
Defaults tuned for 1060-class GPUs: 300 agents, 96x96 map, 200 ticks, logs every 100 ticks.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.kg_manager import KGManager
from Core.FoundationLayer.Foundation.core.world import World
from Core.FoundationLayer.Foundation.wave_mechanics import WaveMechanics
from Core.FoundationLayer.Foundation.world_themes.west_continent.config import WEST_THEME
from Core.FoundationLayer.Foundation.world_themes.west_continent.preset import apply_west_continent_preset


def make_world(map_size: int) -> World:
    kg = KGManager()
    wave = WaveMechanics(kg)
    world = World(primordial_dna={}, wave_mechanics=wave, logger=logging.getLogger("WestSim"))
    apply_west_continent_preset(world, map_size)
    # Perf-friendly toggles
    world.band_split_enabled = True
    world.band_low_resolution = 32
    world.micro_layer_enabled = False
    world.spectrum_log_interval = 200
    world.free_will_threat_threshold = 120.0
    world.free_will_value_threshold = 200.0
    world.trust_scarcity = 0.4
    world.trust_hunger_drain = 0.2
    return world


def spawn_agents(world: World, n: int):
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


def run_sim(agents: int = 300, map_size: int = 96, cycles: int = 200, log_interval: int = 100):
    logging.basicConfig(level=logging.INFO)
    world = make_world(map_size)
    spawn_agents(world, agents)
    logger = world.logger
    logger.info(f"Spawned {agents} agents on {map_size}x{map_size} west continent theme.")

    for t in range(cycles):
        world.run_simulation_step()
        if t % log_interval == 0:
            alive = world.is_alive_mask.sum()
            avg_hp = world.hp[world.is_alive_mask].mean() if alive > 0 else 0.0
            # Simple cooperation/betrayal proxy: coherence/value peaks, threat peak
            threat_peak = float(world.threat_field.max()) if world.threat_field.size else 0.0
            value_peak = float(world.value_mass_field.max()) if world.value_mass_field.size else 0.0
            coh_peak = float(world.coherence_field.max()) if world.coherence_field.size else 0.0
            logger.info(
                f"Tick {t}/{cycles}: alive={alive}, avg_hp={avg_hp:.1f}, "
                f"threat_max={threat_peak:.2f}, value_max={value_peak:.2f}, coherence_max={coh_peak:.2f}"
            )

    logger.info("Simulation complete.")
    return world


def parse_args():
    ap = argparse.ArgumentParser(description="Run West Continent simulation (lightweight).")
    ap.add_argument("--agents", type=int, default=300, help="Number of agents to spawn.")
    ap.add_argument("--map-size", type=int, default=96, help="Map width/height (reduce for speed).")
    ap.add_argument("--cycles", type=int, default=200, help="Simulation ticks.")
    ap.add_argument("--log-interval", type=int, default=100, help="Log interval in ticks.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sim(agents=args.agents, map_size=args.map_size, cycles=args.cycles, log_interval=args.log_interval)
