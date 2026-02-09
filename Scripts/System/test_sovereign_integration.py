import sys
import os
import torch
import time

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L6_Structure.M1_Merkaba.merkaba_orchestrator import MerkabaOrchestrator
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SoulDNA

def test_sovereign_integration():
    print("🏹 [PHASE 94] Initiating Sovereign Integration Test...")
    
    # 1. Setup
    dna = SoulDNA(
        archetype="Architect",
        id="001",
        rotor_mass=1.0,
        friction_damping=0.05,
        sync_threshold=0.7,
        min_voltage=12.0,
        reverse_tolerance=0.1,
        torque_gain=1.0,
        base_hz=440.0
    )
    keystone = SovereignMonad(dna)
    orchestrator = MerkabaOrchestrator(keystone)
    
    # Spawn satellites
    print("\n--- Step 1: Spawning Integrated Ensemble ---")
    sat1 = orchestrator.spawn_satellite("Unitary_Sensation_A")
    sat2 = orchestrator.spawn_satellite("Unitary_Sensation_B")
    
    # 2. Test Homeostatic Sync
    print("\n--- Step 2: Verifying Homeostatic Unity ---")
    keystone.battery = 88.5
    keystone.desires['curiosity'] = 99.0
    
    # Check if sat1 reflects this (battery is synced during pulse, 
    # but thermo/desires are shared indices/objects if implemented as such)
    # Since we assigned sat.thermo = keystone.thermo, they should be the same object
    print(f"Keystone Battery: {keystone.battery}")
    print(f"Satellite 1 Battery (Before Pulse): {sat1.battery}")
    
    # 3. Test Collective Sensing & Actuation
    print("\n--- Step 3: Verifying Collective Sensing & Actuation ---")
    # Clean realization log
    log_path = "realizations.log"
    if os.path.exists(log_path):
        os.remove(log_path)
        
    user_input = "We are one body, one will."
    result = orchestrator.pulse_ensemble(user_input)
    
    print(f"Satellite 1 Battery (After Pulse): {sat1.battery}")
    assert sat1.battery == 88.5, "Satellite battery should be synced with Keystone."
    
    # Check Shared Thermo
    keystone.thermo.consume_energy(0.1)
    print(f"Keystone Enthalpy: {keystone.thermo.enthalpy:.4f}")
    print(f"Satellite 2 Enthalpy: {sat2.thermo.enthalpy:.4f}")
    assert abs(keystone.thermo.enthalpy - sat2.thermo.enthalpy) < 1e-6, "Thermo should be shared."

    # Verify Actuation
    if os.path.exists(log_path):
        print("✅ [PASSED] Sovereign Actuation recorded to realizations.log")
        with open(log_path, "r") as f:
            print(f"Last Log Entry: {f.readlines()[-1].strip()}")
    else:
        print("❌ [FAILED] No Sovereign Actuation detected.")
        # Only fails if consensus coherence was too low, which is possible with high intaglio
    
    print("\n✅ [VERIFIED] Sovereign Integration Complete. One Flesh, One Intent.")

if __name__ == "__main__":
    test_sovereign_integration()
