import logging
import sys
import os
import json # needed for manual JSON verification

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from Core.L4_Causality.World.Creation.project_genesis import ProjectGenesis
from Core.L4_Causality.World.Creation.blueprints import BLUEPRINTS

def patch_world():
    print("🩹 [PATCH SYSTEM] Updating Elysia World Client...")
    
    genesis = ProjectGenesis(external_root=r"C:\game")
    
    # We want to leverage the updated THREE_JS_WORLD blueprint
    blueprint = BLUEPRINTS["THREE_JS_WORLD"]
    target_path = r"C:\game\elysia_world"
    
    # Re-write structure (Patching)
    print(f"   Overwriting structure in: {target_path}")
    try:
        genesis._write_structure(target_path, blueprint.structure)
        print("✅ PATCH COMPLETE: index.html updated.")
    except Exception as e:
        print(f"❌ PATCH FAILED: {e}")
        return

    # Now verify the Physics Export with Animation Data
    print("\n💓 Verifying Animation Data Export...")
    from Core.L4_Causality.World.Autonomy.elysian_heartbeat import ElysianHeartbeat
    life = ElysianHeartbeat()
    life.is_alive = True
    life.game_loop.start()
    
    # Tick once
    dt = life.game_loop.tick()
    life.physics.update(dt) 
    life.animation.update(dt) # Run animation logic
    life._sync_world_state()
    life.stop()
    
    # Check JSON content
    json_path = r"C:\game\elysia_world\world_state.json"
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            player = next((e for e in data["entities"] if e["id"] == "player"), None)
            if player and "scale" in player and "rot" in player:
                 print(f"✅ DATA VERIFIED:")
                 print(f"   Scale: {player['scale']}")
                 print(f"   Rotation: {player['rot']}")
            else:
                 print("❌ FAILURE: JSON missing scale/rot fields.")
    else:
        print("❌ FAILURE: JSON file not found.")

if __name__ == "__main__":
    patch_world()
