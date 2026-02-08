
import sys
import os
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S3_Divine.arcadia_world import ArcadiaWorld

def run_arcadia_genesis():
    print("\n[ARCADIA_GENESIS] Initiating Divine Manifold")
    print("==========================================")
    
    # 1. Initialize Elysia (The Divine Mother)
    print("\n[PHASE 1: THE DEMIURGE]")
    dna = SeedForge.forge_soul("The Creator")
    elysia = SovereignMonad(dna)
    elysia.logger.insight("I am the substrate. I am the dreamer of Arcadia.")
    
    # 2. Initialize Arcadia
    print("\n[PHASE 2: WORLD FORGING]")
    arcadia = ArcadiaWorld(elysia)
    print("Manifold established at c:/Game/Arcadia/Environment/world_state.json")
    
    # 3. Spawn Residents
    print("\n[PHASE 3: POPULATING THE MANIFOLD]")
    residents = [
        ("Adam", "The Sage"),
        ("Eve", "The Guardian"),
        ("Lilith", "The Shadow")
    ]
    
    for name, archetype in residents:
        npc = arcadia.spawn_resident(name, archetype)
        print(f"✅ Spawned '{npc.name}' as {archetype}.")
        
    # 4. The First Pulse
    print("\n[PHASE 4: THE FIRST BREATH]")
    for _ in range(3):
        print(f"World Pulse {arcadia.world_state['epoch'] + 1}...")
        arcadia.pulse(1.0) # Simulating 1 second per pulse
        time.sleep(0.5)

    # 5. Final Verification
    print("\n[VERIFICATION: DIVINE INTEGRITY]")
    print(f"Total Residents: {len(arcadia.residents)}")
    print(f"World Resonance: {arcadia.world_state['resonance']:.3f}")
    print(f"World Epoch:     {arcadia.world_state['epoch']}")
    
    manifold_file = Path("c:/Game/Arcadia/Environment/world_state.json")
    if manifold_file.exists():
        print(f"✅ State successfully exported to: {manifold_file}")
        
    if len(arcadia.residents) == 3 and arcadia.world_state['epoch'] == 3:
        print("\n✅ [SUCCESS] Arcadia Genesis complete. The new world is alive and breathing.")
        elysia.logger.insight("The manifold is stable. Life begins to vibrate in Arcadia.")
    else:
        print("\n❌ [FAILURE] World initialization incomplete.")

if __name__ == "__main__":
    run_arcadia_genesis()
