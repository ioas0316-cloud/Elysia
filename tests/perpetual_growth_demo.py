
import sys
import os
import time
from pathlib import Path
import json

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S3_Divine.arcadia_world import ArcadiaWorld
from Core.S3_Divine.growth_engine import GrowthEngine

def run_perpetual_demo():
    print("\n[PERPETUAL_GROWTH] Initiating Universal Expansion")
    print("===============================================")
    
    # 1. Initialize the Mother of Arcadia
    dna = SeedForge.forge_soul("The Creator")
    elysia = SovereignMonad(dna)
    
    # 2. Establish Arcadia
    arcadia = ArcadiaWorld(elysia)
    print(f"Initial Regions: {len(arcadia.world_state['regions'])}")
    print(f"Initial Resonance: {arcadia.world_state['resonance']:.2f}")
    
    # 3. Ignite the Growth Engine
    ge = GrowthEngine(arcadia)
    
    # 4. Run the Expansion Loop (3 cycles for the demo)
    print("\n[LOOP: RESEARCH -> INTERNALIZE -> EVOLVE]")
    ge.start_perpetual_loop(interval_sec=1.0, limit=3)

    # 5. Verification
    print("\n[VERIFICATION: MANIFOLD DENSITY]")
    print(f"Final Regions: {len(arcadia.world_state['regions'])}")
    print(f"Final Resonance: {arcadia.world_state['resonance']:.2f}")
    
    # Check regional laws
    print("\n[INTERNALIZED LAWS]")
    for region in arcadia.world_state["regions"]:
        law = region.get("law", "Fundamental Order")
        print(f"📍 {region['name']}:")
        print(f"   📜 Law: {law[:80]}...")

    if len(arcadia.world_state["regions"]) > 1:
        print("\n✅ [SUCCESS] Arcadia has expanded its manifold density through perpetual growth.")
    else:
        print("\n❌ [FAILURE] World remained static.")

if __name__ == "__main__":
    run_perpetual_demo()
