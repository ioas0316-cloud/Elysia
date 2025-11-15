import os
import sys

if __name__ == "__main__":
    # Ensure project root on path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from Project_Sophia.core.world import World
    from Project_Sophia.wave_mechanics import WaveMechanics
    from tools.kg_manager import KGManager

    kg = KGManager()
    wm = WaveMechanics(kg)
    world = World(primordial_dna={"instinct": "observe"}, wave_mechanics=wm)

    # Seed a tiny core world: 4 humans starting at 16살
    import numpy as np

    def pos(x, y):
        return {"x": float(x), "y": float(y), "z": 0.0}

    for i in range(4):
        culture = "wuxia" if i % 2 == 0 else "knight"
        world.add_cell(
            f"core_human_{i+1}",
            properties={
                "label": "human",
                "element_type": "animal",
                "culture": culture,
                "gender": "male" if i % 2 == 0 else "female",
                "vitality": 10,
                "wisdom": 8,
                "strength": 9,
                "position": pos(10 + i * 2, 10),
                "age_years": 16,
            },
        )

    print("=== Initial Core World Snapshot ===")
    print("time_step:", world.time_step)
    print("cell_ids:", world.cell_ids)
    print("ages (ticks):", world.age.tolist())

    # Run some steps
    steps = 500
    for _ in range(steps):
        world.run_simulation_step()

    print("\n=== After", steps, "steps ===")
    print("time_step:", world.time_step)
    print("ages (ticks):", world.age.tolist())
    print("hp:", world.hp.tolist())
    print("is_alive_mask:", world.is_alive_mask.tolist())

