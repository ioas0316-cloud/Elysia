import os
import sys
import random
import math


if __name__ == "__main__":
    # Ensure project root on path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from Project_Sophia.core.world import World
    from Project_Sophia.wave_mechanics import WaveMechanics
    from tools.kg_manager import KGManager

    # --- World bootstrap ---
    kg = KGManager()
    wm = WaveMechanics(kg)
    world = World(primordial_dna={"instinct": "observe"}, wave_mechanics=wm)

    # Time scale: 1 tick ≈ 1 day, so 1 year ≈ 365 ticks.
    # This lets us probe long horizons without brute-forcing millions of steps.
    world.set_time_scale(24 * 60.0)

    # For long-horizon survival tuning, start in peaceful mode so we
    # can focus on ecology/survival before full combat/weather.
    world.peaceful_mode = True
    world.macro_food_model_enabled = True

    # --- Seed population: 1,000 humans around the center of the map ---

    def pos(x: float, y: float):
        return {"x": float(x), "y": float(y), "z": 0.0}

    center = world.width / 2.0
    radius = 20.0
    population = 1000

    for i in range(population):
        angle = random.uniform(0.0, 2.0 * math.pi)
        r = radius * (0.3 + 0.7 * random.random())
        x = center + r * math.cos(angle)
        y = center + r * math.sin(angle)

        culture = "wuxia" if i % 2 == 0 else "knight"
        gender = "male" if i % 2 == 0 else "female"

        world.add_cell(
            f"human_{i+1}",
            properties={
                "label": "human",
                "element_type": "animal",
                "culture": culture,
                "gender": gender,
                "vitality": 10,
                "wisdom": 8,
                "strength": 9,
                "position": pos(x, y),
                "age_years": 16,
            },
        )

    print("=== Initial World Snapshot (target: 1,000 humans) ===")
    print("time_step:", world.time_step)
    print("population:", len(world.cell_ids))
    try:
        snapshot = world.get_world_snapshot()
        print("snapshot:", snapshot)
    except Exception as exc:
        print("snapshot: <unavailable>", exc)

    # --- Run approximately one in-world year ---
    years_to_simulate = 1
    year_ticks = world._year_length_ticks()
    steps = years_to_simulate * year_ticks

    for _ in range(steps):
        world.run_simulation_step()

    print("\n=== After ~", years_to_simulate, "simulated year(s) ===")
    print("time_step:", world.time_step)
    print("population:", len(world.cell_ids))
    alive = int(world.is_alive_mask.sum())
    print("alive_count:", alive)
    try:
        snapshot = world.get_world_snapshot()
        print("snapshot:", snapshot)
    except Exception as exc:
        print("snapshot: <unavailable>", exc)
