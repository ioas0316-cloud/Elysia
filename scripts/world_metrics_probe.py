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


def run_probe(steps: int, sample_every: int) -> List[Dict[str, Any]]:
    import numpy as np

    world = make_world()

    # Compress time scale so that short-lived species (e.g., fairies) complete more
    # of their lifespan within a reasonable number of probe steps.
    try:
        world.set_time_scale(100.0)  # 10x fewer ticks per year vs default 10 min/tick
    except Exception:
        pass

    # Simple seed: a small fairy village + plants for food
    rng = np.random.default_rng(42)
    W = getattr(world, "width", 192)
    H = getattr(world, "width", 192)

    # Fairy village center (fae_spring analogue)
    cx = float(W / 2.0)
    cy = float(H / 2.0)

    def rand_village_pos(radius: float = 12.0) -> Dict[str, float]:
        """Random position in a disk around the village center."""
        r = float(rng.uniform(0.0, radius))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        return {
            "x": cx + r * float(np.cos(theta)),
            "y": cy + r * float(np.sin(theta)),
            "z": 0.0,
        }

    # Create a small "spring" of water at the village center for hydration logic.
    if hasattr(world, "wetness"):
        try:
            rad = 3
            x0, x1 = max(0, int(cx) - rad), min(W, int(cx) + rad + 1)
            y0, y1 = max(0, int(cy) - rad), min(H, int(cy) + rad + 1)
            world.wetness[y0:y1, x0:x1] = 1.0
        except Exception:
            pass

    fairy_ids: List[str] = []
    plant_ids: List[str] = []

    # Seed fairies (short-lived humanoids, culturally grouped)
    for i in range(16):
        fairy_id = f"fairy_{i+1}"
        gender = "female" if (i % 2 == 0) else "male"
        world.add_cell(
            fairy_id,
            properties={
                "label": "fairy",
                "element_type": "animal",
                "diet": "omnivore",
                "culture": "fae_village",
                "gender": gender,
                "position": rand_village_pos(radius=10.0),
            },
        )
        fairy_ids.append(fairy_id)

    # Seed plants as simple food sources
    for i in range(40):
        plant_id = f"plant_{i+1}"
        world.add_cell(
            plant_id,
            properties={
                "label": "plant",
                "element_type": "life",
                "position": rand_village_pos(radius=20.0),
            },
        )
        plant_ids.append(plant_id)

    # Light social graph edges to encourage interactions / mating among fairies.
    if fairy_ids:
        for _ in range(len(fairy_ids) * 4):
            a, b = rng.choice(fairy_ids, size=2, replace=False)
            if a != b:
                try:
                    world.add_connection(a, b, float(rng.uniform(0.2, 0.9)))
                except Exception:
                    pass

    # Connect some fairies to nearby plants so herbivorous feeding logic can trigger.
    if fairy_ids and plant_ids:
        for f_id in fairy_ids:
            # Each fairy connects to a small random subset of plants
            for p_id in rng.choice(plant_ids, size=min(5, len(plant_ids)), replace=False):
                try:
                    world.add_connection(f_id, p_id, float(rng.uniform(0.1, 0.3)))
                except Exception:
                    pass

    # Start with high hunger/hydration to avoid immediate collapse
    if world.hunger.size:
        world.hunger[:] = 95.0
    if world.hydration.size:
        world.hydration[:] = 95.0

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
    parser = argparse.ArgumentParser(description="Sample coarse world metrics over time.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=10)
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
