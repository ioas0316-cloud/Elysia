"""Minimal entry point for the new Elysia world package."""

from elysia_world.world import World


def main():
    world = World(primordial_dna={"instinct": "connect_create_meaning"}, wave_mechanics=None)
    world.print_world_summary()


if __name__ == "__main__":
    main()
