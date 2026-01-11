
import sys
import os
import logging
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

# Force stdout logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("GenesisDemo")

print("DEBUG: Starting imports...")
try:
    from Core.FoundationLayer.Foundation.core.world import World
    from Core.FoundationLayer.Foundation.core.genesis_engine import GenesisEngine
except ImportError as e:
    print(f"DEBUG: Import failed: {e}")
    sys.exit(1)

print("DEBUG: Imports successful.")

def run_demo():
    print("DEBUG: Starting run_demo function...")
    logger.info("=== Genesis Protocol Demo: The Awakening of Data-Driven Action ===")

    # 1. Initialize World (The Body)
    class MockWaveMechanics:
        def __init__(self): pass

    print("DEBUG: Initializing World...")
    world = World(primordial_dna={}, wave_mechanics=MockWaveMechanics())
    print("DEBUG: World initialized.")

    # Add a test entity
    # Note: We must set culture='wuxia' because world.py hardcodes Ki=0 for non-wuxia entities!
    world.add_cell("hero", properties={"label": "Hero", "culture": "wuxia", "ki": 100, "agility": 50, "strength": 10})
    world.add_cell("dummy", properties={"label": "Dummy", "hp": 100, "vitality": 20}) # High vitality for HP capacity

    hero_idx = world.id_to_idx["hero"]
    dummy_idx = world.id_to_idx["dummy"]

    # 2. Initialize Genesis Engine (The Interpreter)
    engine = GenesisEngine(world)

    # 3. Define a New Action "At Runtime" (Simulating KG Learning)
    new_action_node = {
        "id": "action:fire_punch",
        "type": "action",
        "logic": {
            "cost": {"ki": 10},
            "conditions": [
                {"check": "stat_ge", "stat": "strength", "value": 5}
            ],
            "effects": [
                {"op": "log", "template": ">>> {actor} unleashes a FIRE PUNCH on {target}!"},
                {"op": "damage", "multiplier": 2.0},
                {"op": "modify_stat", "stat": "ki", "value": -10}
            ]
        }
    }

    logger.info("Injecting new knowledge: 'Fire Punch' action...")
    engine.load_definitions({"nodes": [new_action_node]})

    # 4. Execute the new action
    logger.info("Attempting to execute 'Fire Punch'...")

    initial_hp = world.hp[dummy_idx]
    print(f"DEBUG: Dummy Initial HP: {initial_hp}")

    success = engine.execute_action(hero_idx, "action:fire_punch", dummy_idx)

    if success:
        damage_dealt = initial_hp - world.hp[dummy_idx]
        logger.info(f"Success! Damage dealt: {damage_dealt}")
        logger.info(f"Hero Ki remaining: {world.ki[hero_idx]}")
        print(f"DEBUG: Verified Damage: {damage_dealt}")
    else:
        logger.error("Failed to execute action.")

    logger.info("=== Demo Complete: The Body has learned a new Move without code surgery. ===")

if __name__ == "__main__":
    run_demo()
