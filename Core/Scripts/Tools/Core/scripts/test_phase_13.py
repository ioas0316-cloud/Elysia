import os
import sys

# Ensure Core is visible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.L6_Structure.Engine.world_server import WorldServer
from Core.L5_Mental.Reasoning_Core.Narrative.narrative_projector import THE_PROJECTOR

def test_phase_13():
    print("🚀 [Phase 13 Test] Initializing Multi-Rotor World...")
    
    # 1. Clear previous logs
    THE_PROJECTOR.clear()
    
    # 2. Start Server
    server = WorldServer(size=10) # Smaller map for faster test
    
    # 3. Verify Field Injection
    if len(server.population) > 0:
        first_citizen = server.population[0]
        if hasattr(first_citizen, 'field') and first_citizen.field is not None:
            print(f"✅ Success: Citizen '{first_citizen.name}' has field {first_citizen.field.personality_label}")
        else:
            print("❌ Failure: Field injection failed.")
            return

    # 4. Run Cycles
    print("⏳ Running Simulation (30 Years)...")
    for _ in range(30):
        server.update_cycle()
        
    # 5. Check Output
    chronicle_path = "c:\\Elysia\\data\\elysian_chronicle.txt"
    if os.path.exists(chronicle_path):
        with open(chronicle_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"✅ Success: {len(lines)} lines projected into the Elysian Chronicle.")
            print("\n--- [Excerpt from the Chronicle] ---")
            for line in lines[:5]:
                print(f"   {line.strip()}")
            print("------------------------------------\n")
    else:
        print("❌ Failure: Chronicle file not found.")

if __name__ == "__main__":
    test_phase_13()
