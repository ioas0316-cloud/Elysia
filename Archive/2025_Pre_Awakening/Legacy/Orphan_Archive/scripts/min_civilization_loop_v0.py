import os
import sys
from typing import Dict, Any


def _ensure_repo_root_on_path() -> str:
    """Ensure the repository root is on sys.path and return it."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def _make_pos(x: float, y: float) -> Dict[str, float]:
    return {"x": float(x), "y": float(y), "z": 0.0}


def _seed_min_village(world: "World") -> None:  # type: ignore[name-defined]
    """
    Seed a tiny human village plus surrounding plants directly into the core World.

    Design (CORE_09 · MIN_CIV_LOOP_V0)
    - 20 humans, age_years ~16, mixed gender and culture.
    - 40 plants ("bush" life-type) around them as basic food.
    - No extra starter / UI; pure CellWorld.
    """

    # --- Humans ---
    human_count = 20
    for i in range(human_count):
        gender = "male" if i % 2 == 0 else "female"
        culture = "wuxia" if i % 3 == 0 else "knight"
        props: Dict[str, Any] = {
            "label": "human",
            "element_type": "animal",
            "diet": "omnivore",
            "gender": gender,
            "culture": culture,
            "vitality": 10,
            "wisdom": 8,
            "strength": 9,
            "position": _make_pos(5 + (i % 5) * 2, 5 + (i // 5) * 2),
            "age_years": 16.0,
        }
        world.add_cell(f"village_human_{i+1}", properties=props)

    # --- Plants (simple bushes) ---
    # Increase plant density so that early starvation is less likely.
    plant_count = 120
    for j in range(plant_count):
        px = 2 + (j % 10) * 1.5
        py = 2 + (j // 10) * 1.5
        props = {
            "label": "bush",
            "element_type": "life",
            "diet": "plant",
            "position": _make_pos(px, py),
        }
        world.add_cell(f"village_plant_{j+1}", properties=props)


def run_min_civilization_loop(steps: int = 5000) -> None:
    """
    Run a minimal civilization loop on the core World and print coarse metrics.

    Design intent (MIN_CIV_LOOP_V0 · CORE_09):
    - Time scale: 60 minutes per tick (approx. 24 ticks per in-world day).
    - Duration: 5000 ticks (~208 in-world days under this scale).
    - Question: with basic plants present, do humans survive for a meaningful
      fraction of a year under current laws, or do they collapse quickly?
    """
    _ensure_repo_root_on_path()

    from tools.kg_manager import KGManager
    from Core.FoundationLayer.Foundation.wave_mechanics import WaveMechanics
    from Core.FoundationLayer.Foundation.core.world import World

    kg = KGManager()
    wm = WaveMechanics(kg)
    world = World(
        primordial_dna={
            "instinct": "connect_create_meaning",
            "resonance_standard": "love",
        },
        wave_mechanics=wm,
    )

    # Time scale: 60 minutes per tick (so day_length ~= 24 ticks).
    world.set_time_scale(60.0)

    # Time acceleration: compress the notion of "1 year" so that
    # `steps` ticks ~= target_years of in-world time.
    # Here we want to see if the ecology can persist over ~1000 years
    # (with war/combat disabled).
    target_years = 1000.0
    year_length_days = float(steps) / float(world.day_length * target_years)
    world.year_length_days = year_length_days

    # Peaceful mode: disable lethal combat/weather to measure pure ecological survivability.
    world.peaceful_mode = True

    _seed_min_village(world)

    def summarize(tag: str) -> None:
        snap = world.get_world_snapshot()
        print(f"\n=== {tag} ===")
        print(f"time_step={int(snap['time_step'])}  ~years={snap['approx_years']:.3f}")
        print(f"living humans={int(snap['humans'])}, animals={int(snap['animals'])}, plants={int(snap['plants'])}")
        print(f"avg human age (ticks)={snap['avg_human_age_ticks']:.1f}")
        print(f"avg human hunger={snap['avg_human_hunger']:.1f}, hydration={snap['avg_human_hydration']:.1f}")

        # Simple diagnostic hints for an external observer.
        if snap.get("hint_no_plants", 0.0) > 0:
            print("  hint: no plants remain while humans are alive → long-term 식량 고갈 위험.")
        if snap.get("hint_no_humans", 0.0) > 0 and (snap["plants"] > 0 or snap["animals"] > 0):
            print("  hint: 인간이 사라졌지만 생태계 일부는 남아 있음.")

    summarize("Initial")

    initial_living = world.is_alive_mask.copy()

    for step in range(1, steps + 1):
        world.run_simulation_step()

        if step % 500 == 0 or step == steps:
            summarize(f"After {step} steps")

    # Final simple mortality summary
    final_alive = world.is_alive_mask
    deaths = (~final_alive & initial_living).sum()
    print(f"\nTotal deaths among initial population: {int(deaths)} out of {int(initial_living.sum())}")


if __name__ == "__main__":
    run_min_civilization_loop()
