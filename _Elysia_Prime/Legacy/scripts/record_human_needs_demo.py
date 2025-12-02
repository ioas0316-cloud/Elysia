# [Genesis: 2025-12-02] Purified by Elysia
import json
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
    Demo: run a small world and record human need snapshots per tick.

    Output:
      - logs/world_events.jsonl (from WorldEventLogger)
      - logs/human_needs.jsonl (this script)
    """
    repo_root = _ensure_repo_root_on_path()

    from tools.kg_manager import KGManager
    from Project_Sophia.core.world import World
    from Project_Sophia.wave_mechanics import WaveMechanics
    from ELYSIA.CORE.needs_model import compute_human_needs

    logger = logging.getLogger("needs_demo")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)

    kg_path = os.path.join(repo_root, "data", "kg.json.bak")
    kg_manager = KGManager(filepath=kg_path)
    wm = WaveMechanics(kg_manager=kg_manager)

    world = World(
        primordial_dna={"instinct": "seek_peace_and_nourishment"},
        wave_mechanics=wm,
        logger=logging.getLogger("Project_Sophia.core.world"),
    )

    # Simple setup similar to the small joy scenario.
    world.add_cell(
        "human_demo_1",
        properties={
            "element_type": "animal",
            "label": "human",
            "diet": "omnivore",
            "hp": 35.0,
            "max_hp": 40.0,
        },
    )
    world.add_cell(
        "human_demo_2",
        properties={
            "element_type": "animal",
            "label": "human",
            "diet": "omnivore",
            "hp": 35.0,
            "max_hp": 40.0,
        },
    )

    for i in range(3):
        world.add_cell(
            f"plant_demo_{i}",
            properties={
                "element_type": "life",
                "label": "plant",
                "hp": 80.0,
                "max_hp": 80.0,
            },
        )

    for i in range(3):
        pid = f"plant_demo_{i}"
        world.add_connection("human_demo_1", pid, strength=0.7)
        world.add_connection("human_demo_2", pid, strength=0.7)

    # Prepare needs log.
    os.makedirs("logs", exist_ok=True)
    needs_log_path = os.path.join("logs", "human_needs.jsonl")
    with open(needs_log_path, "w", encoding="utf-8") as f:
        pass

    steps = 50
    logger.info("Running needs demo for %d steps...", steps)
    for _ in range(steps):
        world.run_simulation_step()

        # Record needs for all human-labelled cells.
        with open(needs_log_path, "a", encoding="utf-8") as f:
            for cid, idx in world.id_to_idx.items():
                if idx >= len(world.labels):
                    continue
                if world.labels[idx] != "human":
                    continue
                needs = compute_human_needs(world, idx)
                rec = {
                    "timestamp": world.time_step,
                    "cell_id": cid,
                    "needs": needs.as_dict(),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Needs demo complete. Logs written to logs/world_events.jsonl and %s", needs_log_path)


if __name__ == "__main__":
    main()
