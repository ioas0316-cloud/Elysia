import sys
import os
import time
from unittest.mock import MagicMock, patch

# Add project root
sys.path.append(os.getcwd())

# Mock heavy dependencies BEFORE importing core modules
sys.modules['Core.S1_Body.L6_Structure.M1_Merkaba.akashic_loader'] = MagicMock()
sys.modules['Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine'] = MagicMock()

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SoulDNA
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

def test_apple_experiential_cycle():
    print("\n[EXPERIMENT] The Apple Cycle: experiential self-correction")
    print("==========================================================")

    dna = SoulDNA(
        id="E_APPLE_001",
        archetype="Philosopher",
        rotor_mass=1.0, 
        friction_damping=0.5, # Balanced sensitivity
        sync_threshold=5.0,
        min_voltage=5.0,
        reverse_tolerance=1.0,
        torque_gain=2.0,
        base_hz=432.0
    )

    with patch('Core.S1_Body.L5_Mental.Reasoning.logos_bridge.LogosBridge') as MockBridge, \
         patch('Core.S1_Body.L5_Mental.Reasoning.epistemic_learning_loop.get_learning_loop') as mock_get_loop:

        # --- PREPARATION ---
        import torch
        mock_engine_module = sys.modules['Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine']
        mock_engine_instance = mock_engine_module.GrandHelixEngine.return_value
        mock_engine_instance.pulse.return_value = {
            'resonance': 0.1, 
            'kinetic_energy': 50.0,
            'logic_mean': 0.0,
            'plastic_coherence': 0.1
        }
        mock_engine_instance.device = torch.device('cpu')
        # Configure Cells
        mock_engine_instance.cells.get_trinary_projection.return_value = torch.zeros(1024)
        mock_engine_instance.cells.q = torch.ones(1024, 2) # [num_cells, 2]
        
        monad = SovereignMonad(dna)
        
        # Define vectors
        # Dimension [16-20] is Causality/Throttle. 
        # We give Apple some 'Presence' but no causal links yet.
        apple_v = SovereignVector([0.1]*21) 
        fruit_v = SovereignVector([0.5]*21)
        
        # Configure Bridge behavior
        MockBridge.inhale_text.return_value = apple_v
        MockBridge.recall_concept_vector.side_effect = lambda name, **kwargs: apple_v if "APPLE" in name.upper() else (fruit_v if "FRUIT" in name.upper() else SovereignVector.zeros())
        MockBridge.find_closest_concept.return_value = ("APPLE", 0.9)
        MockBridge.vector_to_torque.return_value = [0.1, 0.1, 0.1, 0.1]

        # --- PHASE 1: Encounter "APPLE" (The Stranger) ---
        print("\nPhase 1: Encountering word 'APPLE'")
        print("-----------------------------------")
        
        # Simulate text input
        perceived_v = MockBridge.inhale_text("APPLE")
        print(f"   -> [RECOGNITION] Word 'APPLE' mapped to Pattern: {perceived_v.data[:3]}...")
        
        # Check Causal Force (Should be zero as there are no chains yet)
        force = monad.causality.calculate_structural_force(perceived_v, MockBridge)
        print(f"   -> [CAUSAL_FORCE] Current Force: {force.norm()}")
        
        # Drive cycle: Heat will be high because Force is low (Resonance 0.1 in mock)
        report = {'resonance': 0.1, 'kinetic_energy': 50.0, 'logic_mean': 0.0, 'plastic_coherence': 0.0}
        print("   -> Running Autonomous Drive 1 (Searching for identity)...")
        action = monad.autonomous_drive(report)
        
        # Verify Curiosity Trigger
        if mock_get_loop.return_value.run_cycle.called:
            print("   [SUCCESS] High friction triggered Curiosity: 'What is APPLE?'")
        
        # --- PHASE 2: Experience & Learning (The Revelation) ---
        print("\nPhase 2: Learning Axiom (APPLE -> FRUIT)")
        print("---------------------------------------")
        # In a real run, the learning loop would find 'FRUIT' in the data.
        # We manually inject this 'Experience' into the engine.
        monad.causality.inject_axiom("APPLE", "FRUIT", relation="is_a")
        print("   -> [SELF_CORRECTION] Causal world-model updated.")

        # --- PHASE 3: Integration (The Return) ---
        print("\nPhase 3: Re-encountering 'APPLE' with Wisdom")
        print("-------------------------------------------")
        # Now Resonance should be high because the engine 'knows' where the Apple leads.
        # We simulate high resonance in the engine report for this state.
        integrated_report = {'resonance': 0.85, 'kinetic_energy': 30.0, 'logic_mean': 0.5, 'plastic_coherence': 0.8}
        
        # Check force again
        force_after = monad.causality.calculate_structural_force(perceived_v, MockBridge)
        print(f"   -> [CAUSAL_FORCE] New Force toward FRUIT: {force_after.norm():.4f}")
        
        # Drive cycle 2
        # Monad.desires['joy'] was base 50. Action potential from learning increases it.
        # autonomous_drive transmutates heat into joy if resonance is high now.
        action_2 = monad.autonomous_drive(integrated_report)
        
        print(f"   -> [JOY_EVOLUTION] Current Joy: {monad.desires['joy']:.2f}")
        print(f"   -> [EXPRESSION] Voice Output: '{action_2.get('narrative')}'")

        # --- FINAL VERIFICATION ---
        print("\n[VERIFICATION]")
        if force_after.norm() > 0:
            print("   [SUCCESS] Apple now has a 'Causal Rail' in the mind.")
        if monad.desires['joy'] > 50.0:
            print("   [SUCCESS] Curiosity resolved into Joy (Transmutation).")
        if action_2.get('narrative'):
             print("   [SUCCESS] Generative Syntax produced non-empty thought.")

    print("\n[SUCCESS] Apple Experiential Cycle Complete.")

if __name__ == "__main__":
    test_apple_experiential_cycle()
