
import sys
import os
import logging
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

# Force stdout logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("AlchemyLab")

from Project_Sophia.core.world import World
from Project_Sophia.core.genesis_engine import GenesisEngine
from Project_Sophia.core.alchemy_cortex import AlchemyCortex

def run_lab():
    logger.info("=== Alchemy Lab: The Concept Synthesizer ===")

    # 1. Setup Lab Environment
    class MockWaveMechanics:
        def __init__(self): pass

    world = World(primordial_dna={}, wave_mechanics=MockWaveMechanics())

    # Subject A: The Monk
    world.add_cell("monk", properties={"label": "Monk", "culture": "wuxia", "ki": 100, "agility": 50, "strength": 20})
    # Subject B: The Dummy
    world.add_cell("dummy", properties={"label": "Dummy", "hp": 200, "vitality": 20})

    monk_idx = world.id_to_idx["monk"]
    dummy_idx = world.id_to_idx["dummy"]

    # 2. Initialize Cortexes
    alchemy = AlchemyCortex()
    genesis = GenesisEngine(world)

    # 3. The Spark: Combining "Wind" and "Punch"
    logger.info("Attempting Synthesis: 'wind' + 'punch'")
    concepts = ["wind", "punch"]

    # Synthesize!
    new_action = alchemy.synthesize_action(concepts)
    logger.info(f"Synthesized DNA: {new_action}")

    # 4. Learning: Load into Genesis Engine
    genesis.load_definitions({"nodes": [new_action]})

    # 5. Execution: Verify Behavior
    action_id = new_action["id"] # action:wind_punch
    logger.info(f"Monk executes '{action_id}' on Dummy...")

    initial_hp = world.hp[dummy_idx]
    initial_agility = world.agility[monk_idx]

    success = genesis.execute_action(monk_idx, action_id, dummy_idx)

    if success:
        dmg = initial_hp - world.hp[dummy_idx]
        agi_change = world.agility[monk_idx] - initial_agility

        logger.info(f"Success!")
        logger.info(f" - Damage Dealt: {dmg} (Base Punch)")
        logger.info(f" - Agility Change: {agi_change} (Wind Effect)")

        if dmg > 0 and agi_change > 0:
            logger.info(">> VALIDATION PASSED: The combined action has properties of both WIND and PUNCH.")
        else:
            logger.warning(">> VALIDATION PARTIAL: Effects missing?")
    else:
        logger.error("Execution Failed.")

if __name__ == "__main__":
    run_lab()
