import logging
import sys
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from Core.L4_Causality.World.Creation.project_genesis import ProjectGenesis

def test_manifestation():
    print("⚔️ [AINCRAD LINK] Initializing ProjectGenesis...")
    
    genesis = ProjectGenesis(external_root=r"C:\game")
    
    project_name = "elysia_world"
    type_key = "THREE_JS_WORLD"
    
    print(f"🏗️ Manifesting '{project_name}' [Type: {type_key}]...")
    
    try:
        success = genesis.create_project(project_name, type_key)
        
        if success:
            print(f"\n✅ SUCCESS: 'Elysia World' manifest at C:\\game\\{project_name}")
            print("   Check 'index.html' for Three.js code.")
            
            # Verify world_state.json creation by running a heartbeat tick
            print("\n💓 simulating Heartbeat for 2 seconds to generate world_state.json...")
            from Core.L4_Causality.World.Autonomy.elysian_heartbeat import ElysianHeartbeat
            life = ElysianHeartbeat()
            life.is_alive = True
            life.game_loop.start()
            
            start_t = time.time()
            while time.time() - start_t < 2.0:
                 # Manually tick
                 dt = life.game_loop.tick()
                 life.idle_time += dt
                 life.physics.update(dt) # Manual update for test script since we aren't using run_loop
                 life._sync_world_state()
                 time.sleep(0.1)
                 
            life.stop()
            
            # Check file
            target_json = r"C:\game\elysia_world\world_state.json"
            if os.path.exists(target_json):
                print(f"✅ CONFIRMED: {target_json} exists and was updated.")
            else:
                print(f"❌ FAILURE: {target_json} was NOT created.")

        else:
            print("\n❌ FAILURE: Project genesis failed.")
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_manifestation()
