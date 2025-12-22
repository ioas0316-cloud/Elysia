
import json
import logging
import os
from elysia_world.world import World
from elysia_world.observer import Observer
from Core.Foundation.wave_mechanics import WaveMechanics

def main():
    """
    Initializes a small world, runs a few steps, and uses the Observer
    to generate and save layered snapshots.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # --- Setup ---
    # Ensure the log directory exists
    log_dir = "elysia_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        logger.info(f"Created directory: {log_dir}")

    class MockKGManager:
        def get_node(self, _):
            return {}

    wave_mechanics = WaveMechanics(kg_manager=MockKGManager())
    world = World(primordial_dna={}, wave_mechanics=wave_mechanics, logger=logger)

    # --- World Population ---
    world.add_cell("human_1", properties={"label": "human", "element_type": "animal", "culture": "wuxia"})
    world.add_cell("tree_1", properties={"label": "tree", "element_type": "life"})
    world.add_cell("wolf_1", properties={"label": "wolf", "element_type": "animal", "diet": "carnivore"})
    logger.info("World initialized with 3 cells.")

    # --- Simulation ---
    for i in range(3):
        logger.info(f"Running simulation step {i+1}...")
        world.run_simulation_step()

    # --- Observation (Layered Snapshots) ---
    observer = Observer(world)

    for level in range(4):
        logger.info(f"--- Generating World Snapshot (Level {level}) ---")
        snapshot = observer.get_world_snapshot(level=level)

        # Print to console for immediate verification
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))

        # Save to file
        output_path = os.path.join(log_dir, f"world_snapshot_level_{level}.json")
        logger.info(f"Saving snapshot to {output_path}")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
            logger.info(f"Level {level} snapshot saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save snapshot for level {level}: {e}")

        logger.info("--------------------------------------------------")

if __name__ == "__main__":
    main()
