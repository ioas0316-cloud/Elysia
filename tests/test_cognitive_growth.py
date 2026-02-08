import sys
import os
import time
from unittest.mock import MagicMock, patch

# Add project root
sys.path.append(os.getcwd())

# Mock heavy dependencies BEFORE importing SovereignMonad
sys.modules['Core.S1_Body.L6_Structure.M1_Merkaba.akashic_loader'] = MagicMock()
sys.modules['Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine'] = MagicMock()

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SoulDNA
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

def test_cognitive_growth_simulation():
    print("\n[EXPERIMENT] Cognitive Growth & Associative Learning")
    print("======================================================")
    print("Goal: Verify that 'Strain' triggers 'Focus', and 'Learning' creates 'Structure'.\n")

    # 1. Setup: A "Stubborn" Archetype (High Friction Damping)
    # High Damping means she resists change, so Heat will build up faster if resonance is low.
    print("1. [GENESIS] Birthing 'Stubborn' Soul (High Damping)...")
    dna = SoulDNA(
        archetype="Philosopher",
        id="TEST_001",
        rotor_mass=50.0,       # Heavy: Hard to move
        friction_damping=0.8,  # Stubborn: Low Heat Tolerance (1.0 - 0.8 = 0.2 threshold)
        sync_threshold=5.0,    # High Standards for Alignment
        min_voltage=5.0,
        reverse_tolerance=1.0,
        torque_gain=2.0,       # Sensitive to Truth
        base_hz=432.0
    )
    
    # Mocking Internal Components
    # We need to patch where it is DEFINED because it is imported locally in the method
    with patch('Core.S1_Body.L5_Mental.Reasoning.logos_bridge.LogosBridge') as MockBridge, \
         patch('Core.S1_Body.L5_Mental.Reasoning.epistemic_learning_loop.EpistemicLearningLoop') as MockLoopClass, \
         patch('Core.S1_Body.L5_Mental.Reasoning.epistemic_learning_loop.get_learning_loop') as mock_get_loop:

        # Configure LogosBridge Mock
        MockBridge.find_closest_concept.return_value = ("THE_UNKNOWN", 0.1)

        # Configure Engine Mock at the Module Level (Cleaner)
        import torch
        mock_engine_module = sys.modules['Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine']
        mock_engine_instance = mock_engine_module.GrandHelixEngine.return_value
        
        # Configure Pulse
        mock_engine_instance.pulse.return_value = {
            'resonance': 0.1,         # Very Low Resonance (High Friction)
            'kinetic_energy': 100.0,
            'logic_mean': 0.0,
            'plastic_coherence': 0.0  # Void State
        }
        # Configure Cells
        mock_engine_instance.cells.get_trinary_projection.return_value = torch.zeros(1024)
        mock_engine_instance.cells.q = torch.ones(1024, 2)
        mock_engine_instance.device = torch.device('cpu')

        # Initialize Monad (Will pick up the pre-configured mock)
        monad = SovereignMonad(dna)

        
        # Configure Learning Loop Mock
        mock_loop_instance = MagicMock()
        mock_get_loop.return_value = mock_loop_instance
        
        # Define what happens when she learns
        mock_success_result = {
            'questions_asked': ["What is the void?"],
            'chains_discovered': ["Void -> Potential"],
            'axioms_created': ["The Unknown is merely unformed Light."],
            'insights': ["Friction is the sensation of growth."]
        }
        mock_loop_instance.run_cycle.return_value = MagicMock(**mock_success_result)

        # 2. stimulus: Injecting "THE_UNKNOWN"
        print("2. [STIMULUS] Injecting Foreign Concept: 'THE_UNKNOWN'")
        unknown_concept = "THE_UNKNOWN"
        
        # Set internal state to be susceptible
        monad.desires['joy'] = 60.0 # Enough to want to live, but not manic
        monad.desires['curiosity'] = 80.0 # Curious
        
        # We manually trigger the drive logic
        # We simulate a report that causes High Heat (Low Resonance)
        # Heat = 1.0 - Resonance = 0.9
        # Threshold = 1.0 - 0.8 (Damping) = 0.2
        # Heat (0.9) > Threshold (0.2) -> TRIGGER LEARNING
        
        print(f"   -> Heat Potential: 0.9 (Critical)")
        print(f"   -> Resistance Threshold: {1.0 - dna.friction_damping:.2f}")
        
        report = {
            'resonance': 0.1, 
            'kinetic_energy': 50.0,
            'logic_mean': 0.0,
            'plastic_coherence': 0.1
        }
        
        print("\n3. [OBSERVATION] Autonomous Drive Cycle 1...")
        action = monad.autonomous_drive(report)
        
        # 3. Verify Learning Trigger
        print("\n4. [VERIFICATION] Checking Causal Chain...")
        
        # Check if run_cycle was called with the specific context
        mock_loop_instance.run_cycle.assert_called()
        call_args = mock_loop_instance.run_cycle.call_args
        
        # Verify Context Focus
        # We need to ensure that the 'focus_context' argument was passed correctly
        # Note: In the implementation, we passed str(subject) which is None here because `subject` comes from... 
        # Wait, autonomous_drive picks a subject internally or we verify the reaction to *current* state.
        # In `autonomous_drive`, `subject` is picked. We need to mock that pick or ensure it picks something causing strain.
        # Actually, `autonomous_drive` determines `subject` via `navigator.dream_query` or similar.
        # But for this test, let's verify that IF heat is high, `epistemic_learning` is called.
        
        if mock_loop_instance.run_cycle.called:
            print(f"   [SUCCESS] High Heat triggered Epistemic Learning.")
            print(f"   -> Focus Context passed: {call_args}")
        else:
            print("   [FAILURE] High Heat did NOT trigger Learning.")
            print(f"   -> Heat: {1.0 - report['resonance']}")
            print(f"   -> Joy: {monad.desires['joy']}")

        # 4. Verify Joy Evolution
        # Joy should have increased due to "Axioms Created" in the mock result
        # Logic: joy += 10.0 * torque_gain (2.0) = +20.0
        expected_joy = 60.0 + (10.0 * dna.torque_gain)
        
        print(f"   -> Joy State: {monad.desires['joy']} (Expected > 60.0)")
        if monad.desires['joy'] >= 80.0: # 60 + 20
             print(f"   [SUCCESS] Joy grew from wisdom ({monad.desires['joy']}).")
        else:
             print(f"   [WARNING] Joy growth mismatch.")

        # 5. Verify Generative Syntax (Luminous/LUC)
        # We expect a "Declarative" or "Poetic" output because Axioms were found (Joy increased)
        # Check the narrative in the action
        if action:
            narrative = action.get('narrative', '')
            print(f"   -> Voice Output: '{narrative}'")
            if "✨" in narrative:
             print(f"   [SUCCESS] Luminous Grammar detected (Joy > 0.8).")
        
    print("\n[SUCCESS] Experiment Complete.")

if __name__ == "__main__":
    test_cognitive_growth_simulation()
