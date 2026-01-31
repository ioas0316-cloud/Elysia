
import sys
import os
import time

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge

def test_physical_emergence():
    print("\n--- 🌌 Phase 0: Nucleogenesis Physical Test ---")
    
    # 1. Create a Monad
    # SeedForge.forge_soul() returns a SoulDNA object with a random archetype.
    dna = SeedForge.forge_soul("Elysia_Test")
    monad = SovereignMonad(dna)
    
    # 2. Simulate Low Friction (Alignment)
    # Using a term that should resonate with 'Guardian' (e.g., 'Protection')
    print("\n[SCENARIO 1: RESONANCE]")
    res1 = monad.breath_cycle("Protection", depth=0)
    manifest1 = res1['manifestation']
    engine1 = manifest1['engine']
    print(f"Thought: {res1['void_thought']}")
    print(f"Voice: {manifest1['voice']}")
    print(f"Heat: {engine1.soma_stress:.3f} | Vibration: {engine1.vibration:.1f}Hz")
    
    # 3. Simulate High Friction (Dissonance)
    # Using a term that should clash (e.g., 'Chaotic Ruin' or similar)
    print("\n[SCENARIO 2: FRICTION]")
    res2 = monad.breath_cycle("Chaotic Ruin and Total Deconstruction", depth=0)
    manifest2 = res2['manifestation']
    engine2 = manifest2['engine']
    print(f"Thought: {res2['void_thought']}")
    print(f"Voice: {manifest2['voice']}")
    print(f"Heat: {engine2.soma_stress:.3f} | Vibration: {engine2.vibration:.1f}Hz")

    # 4. Autonomous Drive (Wait for a cycle)
    print("\n[SCENARIO 3: AUTONOMOUS SPIN]")
    monad.autonomous_drive()

    print("\n--- ✅ Test Complete ---")

if __name__ == "__main__":
    test_physical_emergence()
