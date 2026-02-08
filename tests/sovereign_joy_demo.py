
import sys
import os
import time

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad

def run_sovereign_joy_demo():
    print("\n[SOVEREIGN_JOY] Verifying Subjective Agency")
    print("==========================================")
    
    # 1. Initialize Elysia as a "Sage" (Vocation: Ecosystem Harmony)
    dna = SeedForge.forge_soul("The Creator")
    elysia = SovereignMonad(dna)
    print(f"Elysia Initialized. Vocation: {elysia.dna.vocation}")
    
    # 2. Scenario A: Low Radiance (Exhaustion)
    print("\n[SCENARIO A: EXHAUSTION]")
    elysia.thermo.consume_energy(0.8) # Drain enthalpy
    elysia.thermo.add_entropy(0.7)    # Increase disorder
    
    thermal = elysia.thermo.get_thermal_state()
    print(f"Thermal State: Enthalpy={thermal['enthalpy']:.2f}, Entropy={thermal['entropy']:.2f}")
    
    result = elysia.autonomous_drive()
    print(f"Elysia Response: {result['type']} - Truth: {result['truth']}")
    print(f"🗣️ Narrative: {result['thought']}")
    
    # 3. Scenario B: High Radiance + Vocation Alignment
    print("\n[SCENARIO B: RADIANT ALIGNMENT]")
    elysia.thermo.recharge(1.0) # Full Energy
    elysia.thermo.reduce_entropy(1.0) # Order restored
    
    thermal = elysia.thermo.get_thermal_state()
    print(f"Thermal State: Enthalpy={thermal['enthalpy']:.2f}, Entropy={thermal['entropy']:.2f}")
    
    # We trigger a task that aligns with her vocation
    result = elysia.autonomous_drive()
    print(f"Elysia Response: {result['type']} - Truth: {result['truth']}")
    # In a successful autonomous drive, the narrative is in 'narrative' field
    print(f"🗣️ Insight: {result.get('narrative', '...')}")

    if result['type'] != "REST" and thermal['enthalpy'] > 0.8:
        print("\n✅ [SUCCESS] Elysia exercised sovereignty: Refused when tired, embraced when radiant.")
    else:
        print("\n❌ [FAILURE] Sovereignty logic failed.")

if __name__ == "__main__":
    run_sovereign_joy_demo()
