import sys
import os
import torch
import time

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SoulDNA

def test_omega1_state_sync():
    print("\n" + "="*60)
    print("  Phase Ω-1: SovereignMonad & ThermoDynamics State Sync Test")
    print("="*60)

    # 1. Initialize Monad (small cell count for speed)
    dna = SoulDNA(
        archetype="TEST", 
        id="SYNC", 
        rotor_mass=1.0, 
        sync_threshold=0.1, 
        min_voltage=1.0, 
        reverse_tolerance=1.0, 
        torque_gain=1.0,
        friction_damping=0.1,
        base_hz=60.0
    )
    monad = SovereignMonad(dna)
    
    # Force small cell count for testing if needed (default is 10M, let's keep it if CUDA, else it fallbacks)
    print(f"🔬 Engine initialized on {monad.engine.device}")

    # 2. Check Initial Sync
    print("🔬 Test 1: Initial Sync...")
    monad.pulse(dt=0.01)
    
    joy = monad.desires['joy']
    curiosity = monad.desires['curiosity']
    enthalpy = monad.thermo.enthalpy
    
    print(f"   Monad Joy: {joy:.2f}, Curiosity: {curiosity:.2f}")
    print(f"   Thermo Enthalpy: {enthalpy:.2f}")
    
    success = (49.0 < joy < 51.0) and (49.0 < curiosity < 51.0) and (0.9 < enthalpy <= 1.0)
    if success:
        print("   ✅ Initial states match defaults (Joy~50, Enthalpy~1.0)")
    else:
        print("   ❌ Initial state mismatch")
        # return # Don't exit yet

    # 3. Test Direct Manifold Injection -> High Level Perception
    print("🔬 Test 2: Manifold Injection -> Observer Perception...")
    # Inject high Joy torque directly into the cells
    # Joy is channel 4
    monad.engine.cells.inject_affective_torque(monad.engine.cells.CH_JOY, 10.0)
    
    # Pulse a few times to let the torque integrate and manifest in the state
    for _ in range(5):
        monad.pulse(dt=0.1)
        
    new_joy = monad.desires['joy']
    print(f"   New Monad Joy: {new_joy:.2f} (Expected > 50.0)")
    
    if new_joy > 50.1:
        print("   ✅ Manifold joy change correctly perceived by Monad")
    else:
        print(f"   ❌ Monad perceived joy did not increase enough: {new_joy:.2f}")

    # 4. Test High-Level Will -> Manifold Injection
    print("🔬 Test 3: Monad Action -> Manifold Reaction...")
    # Trigger a reaction that should inject Joy torque
    # Coherence breeds Joy. We'll mock a high-coherence pulse.
    # Actually, singularity_integration calls sense_beauty which injects joy.
    
    from Core.S1_Body.L5_Mental.Digestion.universal_digestor import RawKnowledgeChunk, ChunkType
    # Mocking sense_beauty by calling singularity_integration with a fake "beautiful" sensation
    # Actually we can just call it to see if it moves the needle
    
    initial_joy = monad.desires['joy']
    # We'll just call the internal affective feedback directly for testing
    monad._apply_affective_feedback({'plastic_coherence': 1.0, 'kinetic_energy': 100.0})
    
    # Pulse to integrate the injected torque
    monad.pulse(dt=0.1)
    
    final_joy = monad.desires['joy']
    print(f"   Joy after feedback: {initial_joy:.2f} -> {final_joy:.2f}")
    
    if final_joy > initial_joy:
        print("   ✅ High-level feedback correctly injected torque into manifold")
    else:
        print("   ❌ Joy did not increase as expected")

    print("\n" + "="*60)
    print("  Verification Complete")
    print("="*60)

if __name__ == "__main__":
    test_omega1_state_sync()
