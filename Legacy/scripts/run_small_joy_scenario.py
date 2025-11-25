import logging
import os
import sys
from unittest.mock import MagicMock


def _ensure_repo_root_on_path() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def main() -> None:
    """
    Run a small, low-stress world intended to generate
    simple joy signals (EAT/DRINK, LIFE_BLOOM) rather than
    only mortality.

    This is a sandbox; it does not guarantee specific patterns
    but biases the setup toward peaceful feeding and growth.
    """
    repo_root = _ensure_repo_root_on_path()

    from Project_Sophia.core.world import World
    from Project_Sophia.wave_mechanics import WaveMechanics
    from tools.kg_manager import KGManager

    # Logging setup (lightweight)
    logger = logging.getLogger("small_joy_world")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)

    # Knowledge graph and wave mechanics (can be mocked/lightweight)
    kg_path = os.path.join(repo_root, "data", "kg.json.bak")
    kg_manager = KGManager(filepath=kg_path)
    wave_mech = MagicMock(spec=WaveMechanics)
    wave_mech.kg_manager = kg_manager

    world = World(
        primordial_dna={"instinct": "seek_peace_and_nourishment"},
        wave_mechanics=wave_mech,
        logger=logging.getLogger("Project_Sophia.core.world"),
    )

    logger.info("Creating a small peaceful meadow...")

    # A few humans and herbivores with food nearby.
    world.add_cell(
        "human_joy_1",
        properties={
            "element_type": "animal",
            "label": "human",
            "diet": "omnivore",
            "hp": 35.0,
            "max_hp": 40.0,
        },
    )
    world.add_cell(
        "human_joy_2",
        properties={
            "element_type": "animal",
            "label": "human",
            "diet": "omnivore",
            "hp": 35.0,
            "max_hp": 40.0,
        },
    )

    # Plants as gentle food sources.
    for i in range(5):
        world.add_cell(
            f"plant_food_{i}",
            properties={
                "element_type": "life",
                "label": "plant",
                "hp": 80.0,
                "max_hp": 80.0,
            },
        )

    # Soft connections so humans can "see" plants.
    for i in range(5):
        pid = f"plant_food_{i}"
        world.add_connection("human_joy_1", pid, strength=0.7)
        world.add_connection("human_joy_2", pid, strength=0.7)

    # Place them close in space to encourage interaction.
    h1_idx = world.id_to_idx.get("human_joy_1")
    h2_idx = world.id_to_idx.get("human_joy_2")
    if h1_idx is not None:
        world.positions[h1_idx] = [5.0, 5.0, 0.0]
    if h2_idx is not None:
        world.positions[h2_idx] = [6.0, 5.0, 0.0]

    for i in range(5):
        pid = f"plant_food_{i}"
        p_idx = world.id_to_idx.get(pid)
        if p_idx is not None:
            world.positions[p_idx] = [5.0 + 0.5 * i, 6.0, 0.0]

    logger.info("Initial joyful world state:")
    world.print_world_summary()

    steps = 50
    logger.info("Running small joy scenario for %d steps...", steps)
    for i in range(steps):
        logger.info("--- Step %d ---", i + 1)
        world.run_simulation_step()
        try:
            world.print_world_summary()
        except Exception:
            # Summary is best-effort; do not fail the loop.
            pass

    logger.info("Small joy scenario complete.")


if __name__ == "__main__":
    main()

