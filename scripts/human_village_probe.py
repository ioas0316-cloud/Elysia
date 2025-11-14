import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any


def make_world() -> "World":
    """Construct a minimal Project_Sophia.core.world.World instance."""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tools.kg_manager import KGManager
    from Project_Sophia.wave_mechanics import WaveMechanics
    from Project_Sophia.core.world import World

    kgm = KGManager()
    wm = WaveMechanics(kg_manager=kgm)
    world = World(
        primordial_dna={
            "instinct": "connect_create_meaning",
            "resonance_standard": "love",
        },
        wave_mechanics=wm,
    )
    return world


def seed_human_village(world: "World", rng) -> Dict[str, Any]:
    """
    Seed a small human village around a central well, following WORLD_KIT_HUMAN_VILLAGE v0.

    - 12~20 humans clustered near home_pos (village_1)
    - A 'well' represented as high wetness at home_pos
    - A nearby demo_field of plants with slightly denser coverage
    """
    import numpy as np

    W = getattr(world, "width", 256)
    H = getattr(world, "width", 256)

    # Village center = "home_pos" (well + plaza)
    cx = float(W / 2.0)
    cy = float(H / 2.0)
    home_pos = (cx, cy)

    # Mark a small well area via wetness, so hydration field will attract thirsty agents.
    if hasattr(world, "wetness"):
        try:
            rad = 2
            x0, x1 = max(0, int(cx) - rad), min(W, int(cx) + rad + 1)
            y0, y1 = max(0, int(cy) - rad), min(H, int(cy) + rad + 1)
            world.wetness[y0:y1, x0:x1] = 1.0
        except Exception:
            pass

    def rand_near_home(radius: float = 10.0) -> Dict[str, float]:
        r = float(rng.uniform(0.0, radius))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        return {
            "x": cx + r * float(np.cos(theta)),
            "y": cy + r * float(np.sin(theta)),
            "z": 0.0,
        }

    def rand_demo_field_pos(side: str = "left", size: int = 10) -> Dict[str, float]:
        """Place plants in a 10x10-ish demo field near the village."""
        if side == "left":
            base_x = cx - 16.0
        else:
            base_x = cx + 16.0
        base_y = cy
        x = base_x + float(rng.uniform(-size / 2.0, size / 2.0))
        y = base_y + float(rng.uniform(-size / 2.0, size / 2.0))
        return {"x": x, "y": y, "z": 0.0}

    human_ids: List[str] = []
    plant_ids: List[str] = []

    # Seed humans: 12~20 adults, culture='village'
    n_humans = int(rng.integers(12, 21))
    for i in range(n_humans):
        hid = f"human_{i+1}"
        gender = "female" if (i % 2 == 0) else "male"
        # Age range in years ~ [18, 30]; world will convert using year_length_ticks internally.
        age_years = int(rng.integers(18, 31))
        world.add_cell(
            hid,
            properties={
                "label": "human",
                "element_type": "animal",
                "diet": "omnivore",
                "culture": "village",
                "gender": gender,
                "age": age_years * max(1, int(world._year_length_ticks() / max(1, world.year_length_days))),  # approx
                "position": rand_near_home(radius=10.0),
            },
        )
        human_ids.append(hid)

    # Seed demo_field plants near the village (fast-visible food/greenery)
    for i in range(40):
        pid = f"field_plant_{i+1}"
        side = "left" if (i % 2 == 0) else "right"
        world.add_cell(
            pid,
            properties={
                "label": "plant",
                "element_type": "life",
                "position": rand_demo_field_pos(side=side, size=10),
            },
        )
        plant_ids.append(pid)

    # Light social graph: connect humans to each other (village relations)
    if human_ids:
        for _ in range(len(human_ids) * 4):
            a, b = rng.choice(human_ids, size=2, replace=False)
            if a != b:
                try:
                    world.add_connection(a, b, float(rng.uniform(0.2, 0.9)))
                except Exception:
                    pass

    # Connect some humans to plants so herbivorous feeding logic can trigger if needed.
    if human_ids and plant_ids:
        for h_id in human_ids:
            for p_id in rng.choice(plant_ids, size=min(5, len(plant_ids)), replace=False):
                try:
                    world.add_connection(h_id, p_id, float(rng.uniform(0.1, 0.3)))
                except Exception:
                    pass

    return {
        "home_pos": home_pos,
        "human_ids": human_ids,
        "plant_ids": plant_ids,
    }


def run_probe(steps: int, sample_every: int) -> List[Dict[str, Any]]:
    import numpy as np

    world = make_world()

    # Use a calmer time scale for human village; default 10 minutes/tick is fine.
    # (No change needed unless we want to see multi-decade dynamics.)

    rng = np.random.default_rng(1234)
    seed_info = seed_human_village(world, rng)

    # Start with fairly high hunger/hydration to avoid instant collapse.
    if world.hunger.size:
        world.hunger[:] = 90.0
    if world.hydration.size:
        world.hydration[:] = 90.0

    samples: List[Dict[str, Any]] = []

    for _ in range(steps):
        try:
            world.run_simulation_step()
        except TypeError:
            world.run_simulation_step()

        if world.time_step % sample_every == 0:
            try:
                m = world.get_population_metrics()
            except Exception:
                m = {"time_step": int(world.time_step)}
            samples.append(m)

        if len(getattr(world, "cell_ids", [])) == 0:
            break

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a small human village around a well.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--sample-every", type=int, default=20)
    args = parser.parse_args()

    samples = run_probe(args.steps, args.sample_every)

    # Simple CSV-like output for quick inspection / redirection to file
    header = [
        "time_step",
        "living",
        "animals",
        "plants",
        "humans",
        "fairies",
        "hunger_animals_mean",
        "hunger_humans_mean",
        "hunger_fairies_mean",
        "hydration_animals_mean",
        "hydration_humans_mean",
        "hydration_fairies_mean",
    ]
    print(",".join(header))
    for m in samples:
        row = [
            str(m.get("time_step", 0)),
            str(m.get("living", 0)),
            str(m.get("animals", 0)),
            str(m.get("plants", 0)),
            str(m.get("humans", 0)),
            str(m.get("fairies", 0)),
            f"{m.get('hunger_animals_mean', 0.0):.2f}",
            f"{m.get('hunger_humans_mean', 0.0):.2f}",
            f"{m.get('hunger_fairies_mean', 0.0):.2f}",
            f"{m.get('hydration_animals_mean', 0.0):.2f}",
            f"{m.get('hydration_humans_mean', 0.0):.2f}",
            f"{m.get('hydration_fairies_mean', 0.0):.2f}",
        ]
        print(",".join(row))


if __name__ == "__main__":
    main()

