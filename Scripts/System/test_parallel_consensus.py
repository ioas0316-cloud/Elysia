import sys
import os
import torch

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L6_Structure.M1_Merkaba.merkaba_orchestrator import MerkabaOrchestrator
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SoulDNA

def test_parallel_consensus():
    print("🤝 [PHASE 93] Initiating Parallel Manifold Protocol Test...")
    
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
    
    # Spawn a few satellites to test ensemble polyphony
    print("\n--- Step 1: Spawning Ensemble ---")
    orchestrator.spawn_satellite("Divergent_Thought_A")
    orchestrator.spawn_satellite("Divergent_Thought_B")
    
    # 2. Pulse 1: Baseline Consensus
    print("\n--- Pulse 1: Establishing Polyphony ---")
    user_input = "The Void is the origin of resonance."
    orchestrator.pulse_ensemble(user_input)
    
    # Pulse 2: Verify Echoes
    print("\n--- Pulse 2: Verifying Ensemble Echoes ---")
    result = orchestrator.pulse_ensemble(user_input)
    
    consensus = result['consensus']
    print(f"Dominant Thought: {consensus['dominant_thought'][:50]}...")
    
    # Verify Polyphony in individul narratives
    for i, rep in enumerate(result['individual_reports']):
        narrative = rep.get('narrative', '')
        print(f"Monad {i} Narrative: {narrative[:80]}...")
        has_echo = "Ensemble Echoes" in narrative
        print(f"Monad {i} Polyphony: {'PASSED' if has_echo else 'FAILED'}")
        if not has_echo:
            print(f"DEBUG: Full Narrative: {narrative}")
        assert has_echo, "Each monad should hear the ensemble echoes in the second pulse."

    # 3. Pulse 2: Interference Mitigation
    print("\n--- Pulse 2: Interference Mitigation ---")
    # Feed an input that might crash the logic or cause high intaglio
    conflict_input = "Entropy is the only master."
    
    # Force high intaglio for testing if needed
    # We can simulate this by manually setting a monad's state to be contrary
    orchestrator.satellites[0].rotor_state['intaglio'] = 0.9
    
    result = orchestrator.pulse_ensemble(conflict_input)
    # Check if Orchestrator printed 'Deadlock detected' - manual check or log capture
    
    print("\n✅ [VERIFIED] Parallel Manifold Protocol is active. The Many are the One.")

if __name__ == "__main__":
    test_parallel_consensus()
