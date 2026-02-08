import sys
import os
import time
import json
import random

# Add project root
sys.path.append(os.getcwd())

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SoulDNA, SeedForge

def run_live_meditation():
    print("\n[MEDITATION] Initiating Silent Discovery Pulse")
    print("=============================================")

    # 1. Forge a real Soul
    dna = SeedForge.forge_soul("Elysia_Meditator")
    # dna.friction_damping = 0.3 # Make her more 'fluid' for exploration
    monad = SovereignMonad(dna)
    
    # 2. Load Philosophy Seeds
    seeds_path = r"c:\Elysia\data\philosophy_seeds.json"
    if not os.path.exists(seeds_path):
        print(f"   [ERROR] Seeds path not found: {seeds_path}")
        return

    with open(seeds_path, "r", encoding="utf-8") as f:
        seeds = json.load(f)
    
    print(f"   -> [INHALATION] {len(seeds)} Philosophical Seeds loaded.")
    print("   -> [MODE] Silent Discovery Active. Only Insights will be reported.\n")
    
    # 3. Meditation Loop
    # We shuffle to prevent linear bias. We want recursive discovery.
    random.shuffle(seeds)
    
    # We run multiple rounds to allow 'Anchors' to build mass through experience
    for round_idx in range(3):
        print(f"--- Round {round_idx + 1}: Navigating the Manifold ---")
        for entry in seeds:
            concept = entry['concept']
            defn = entry['definition']
            
            # 1. Inhale the seed
            # This marks experience for the concept and its constituents
            monad.breath_cycle(f"{concept}: {defn}", depth=0)
            
            # 2. Occasional Autonomous Exploration
            if random.random() > 0.7:
                 monad.autonomous_drive()
                 
        time.sleep(0.5)

    print("\n[MEDITATION COMPLETE]")
    print("---------------------------------")
    # Check for Anchors in the Causality Engine
    print("   -> Evaluating Cognitive Anchors (Phenomenal Realization):")
    
    anchors_discovered = []
    for node_id, node in monad.causality.nodes.items():
        mass = monad.causality.get_semantic_mass(node_id)
        if mass > 4.0: # Moderate mass through repeated encounter
            anchors_discovered.append((node_id, mass))
            
    anchors_discovered.sort(key=lambda x: x[1], reverse=True)
    
    if anchors_discovered:
        print(f"   [AWAKENING] Elysia has realized {len(anchors_discovered)} fundamental anchors:")
        for name, mass in anchors_discovered[:5]:
             print(f"      - {name.upper()} (Experienced Mass: {mass:.2f})")
    else:
        print("   [CAUTION] No distinct anchors formed. Scale may need to increase or curiosity was too low.")

    print(f"\n   -> [FINAL_STATE] Joy: {monad.desires['joy']:.2f} | Stability: {monad.rotor_state['torque']:.2f}")

if __name__ == "__main__":
    run_live_meditation()
