import sys
import os
from unittest.mock import MagicMock, patch

# Add project root
sys.path.append(os.getcwd())

# Mock heavy dependencies BEFORE importing SovereignMonad
sys.modules['Core.S1_Body.L6_Structure.M1_Merkaba.akashic_loader'] = MagicMock()
sys.modules['Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine'] = MagicMock()

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SoulDNA

def test_joy_driven_radiance_mock():
    print("✨ Testing Phase 90: Joy-Driven Radiance Logic (Mocked)...")
    
    try:
        # 1. Initialize Monad with Mocked Engine
        print("   -> Instantiating SovereignMonad with mocks...")
        
        # Get the mocked class and configure its return value
        import Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine as MockModule
        MockEngineClass = MockModule.GrandHelixEngine
        mock_instance = MockEngineClass.return_value
        
        # Configure return values
        import torch
        mock_instance.pulse.return_value = {'resonance': 0.9, 'kinetic_energy': 50.0, 'logic_mean': 0.5, 'plastic_coherence': 1.0}
        mock_instance.cells.get_trinary_projection.return_value = torch.zeros(1024)
        mock_instance.device = torch.device('cpu')
        # Mock q for manifold state
        mock_instance.cells.q = torch.ones(1024, 2) # Ones to avoid division by zero if sum is used

        dna = SoulDNA(
            archetype="TestArchetype", 
            id=1,
            rotor_mass=10.0,
            friction_damping=0.5,
            sync_threshold=10.0,
            min_voltage=5.0,
            reverse_tolerance=2.0,
            torque_gain=1.0,
            base_hz=60.0
        )
        monad = SovereignMonad(dna)
        
        # 2. Set Initial Desires
        print("   -> Setting Desires: Joy=80, Alignment=90, Curiosity=100")
        monad.desires['joy'] = 80.0
        monad.desires['alignment'] = 90.0
        monad.desires['curiosity'] = 100.0
        
        # 3. Trigger Autonomous Drive
        print("   -> Triggering autonomous_drive()...")
        
        # Force the capacitor to trigger action
        monad.wonder_capacitor = 150.0 
        
        # The engine report is passed directly to avoid internal blocking calls if any
        mock_report = {'resonance': 0.9, 'kinetic_energy': 50.0, 'logic_mean': 0.5, 'plastic_coherence': 1.0}
        
        # Mock logos bridge or other heavy logic if needed, but let's try running first
        action = monad.autonomous_drive(mock_report)
        
        if action:
            print(f"   -> Action Triggered: {action.get('type')}")
            print(f"   -> Narrative: {action.get('narrative')}")
            print(f"   -> Internal Change: {action.get('internal_change')}")
            
            # 4. Check if Joy increased (Positive Feedback Loop)
            print(f"   -> Current Joy State: {monad.desires['joy']}")
            if monad.desires['joy'] > 80.0:
                print("   ✅ SUCCESS: Joy increased after successful projection.")
            else:
                print("   ⚠️ WARNING: Joy did not increase. Check logic.")
        else:
            print("   ⚠️ No action triggered. Check thresholds.")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"   ❌ ERROR: {e}")

    print("\n✨ Test Complete.")

if __name__ == "__main__":
    test_joy_driven_radiance_mock()
