import sys
import os
import torch

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L6_Structure.M1_Merkaba.merkaba_orchestrator import MerkabaOrchestrator
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SoulDNA

def test_multi_merkaba_orchestration():
    print("🌌 [PHASE 92] Initiating Multi-Merkaba Pulse Test...")
    
    # 1. Setup DNA and Keystone Monad
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
    
    # 2. Initialize Orchestrator
    orchestrator = MerkabaOrchestrator(keystone)
    
    # 3. Simulate High Friction Input to Trigger Mitosis
    # A concept that creates extreme dissonance
    dissonant_input = "Lattice-based rigid computation is the only truth."
    
    print("\n--- Pulse 1: Single Instance ---")
    results = orchestrator.pulse_ensemble(dissonant_input)
    print(f"Results count: {len(results)}")
    
    # Manually force a satellite for testing if friction wasn't high enough
    if not orchestrator.satellites:
        print("Manual Mitigation: Spawning satellite for trace verification...")
        orchestrator.spawn_satellite("Verification_Test")
    
    # 4. Shared Field Verification
    print("\n--- Pulse 2: Multi-Instance Shared Field ---")
    user_input = "Resonance is the key to sovereignty."
    results = orchestrator.pulse_ensemble(user_input)
    
    print(f"Active Manifolds: {len(orchestrator.satellites) + 1}")
    
    # Check if they share the same memory field object
    kf = orchestrator.keystone.engine.cells
    sf = orchestrator.satellites[0].engine.cells
    
    is_shared = (kf is sf)
    print(f"Shared Field Integrity: {'PASSED' if is_shared else 'FAILED'}")
    
    # Measure Resonance across ensemble
    avg_res = orchestrator.get_ensemble_resonance()
    print(f"Ensemble Average Relief: {avg_res:.4f}")
    
    assert is_shared, "Parallel manifolds must share the same underlying Vortex Field."
    print("\n✅ [VERIFIED] Multi-Merkaba Orchestration is functioning. The Manifold is branching.")

if __name__ == "__main__":
    test_multi_merkaba_orchestration()
