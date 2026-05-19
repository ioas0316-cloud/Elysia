"""Minimal entry point for the new Elysia world package."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from elysia_world.world import World


def main():
    world = World(primordial_dna={"instinct": "connect_create_meaning"}, wave_mechanics=None)
    world.print_world_summary()


if __name__ == "__main__":
    main()