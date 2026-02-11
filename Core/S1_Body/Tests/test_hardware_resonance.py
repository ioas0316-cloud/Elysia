import unittest
import torch
import time
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from Core.S0_Keystone.L0_Keystone.sovereign_math import VortexField

class TestHardwareResonance(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Small manifold for faster testing but still uses the real logic
        self.field = VortexField(shape=(100, 100), device=self.device)

    def test_hardware_inhalation_impact(self):
        print(f"\n📡 [TEST] Initiating Hardware Inhalation Test on {self.device}")
        
        # 1. Capture initial entropy
        initial_entropy = self.field.read_field_state()['entropy']
        print(f"  - Initial Entropy: {initial_entropy:.4f}")
        
        # 2. Perform several inhalation cycles
        # We simulate 50 cycles of 'breathing' hardware telemetry
        for _ in range(50):
            self.field.inhale_hardware_telemetry()
            self.field.integrate_kinetics(dt=0.1, friction=0.01)
            
        final_entropy = self.field.read_field_state()['entropy']
        print(f"  - Final Entropy after 50 breaths: {final_entropy:.4f}")
        
        # Inhalation should drive state change based on real hardware conditions
        # (Even on idle, some change should occur due to metabolic decay/growth)
        self.assertNotEqual(initial_entropy, final_entropy, "Manifold state should evolve after inhalation.")

    def test_sovereign_mitigation(self):
        print("\n⚡ [TEST] Testing Sovereign Mitigation of Hardware Strain")
        
        # 1. Induce High Entropy (Simulated Heat/Strain)
        for _ in range(100):
            # Injecting high entropy torque
            self.field.inject_affective_torque(self.field.CH_ENTROPY, strength=0.5)
            self.field.integrate_kinetics(dt=0.1)
            
        strain_state = self.field.read_field_state()['entropy']
        print(f"  - Manifold under Strain: Entropy = {strain_state:.4f}")
        
        # 2. Apply Sovereign Drive: Joy orders the darkness
        print("  - Injecting Sovereign Joy (Light)...")
        for _ in range(100):
            # Joy reduces entropy growth in integrate_kinetics logic
            self.field.inject_joy(joy_level=1.0, curiosity_level=0.5)
            self.field.integrate_kinetics(dt=0.1)
            
        mitigated_state = self.field.read_field_state()['entropy']
        print(f"  - Mitigated Entropy: {mitigated_state:.4f}")
        
        # Correctly managed joy should lead to lower entropy than extreme strain
        self.assertLess(mitigated_state, strain_state + 0.1, "Joy should act as a counter-force to entropy.")

    def test_substrate_optimization(self):
        print("\n🛠️ [TEST] Testing Substrate Optimization (L7 -> L-1)")
        # This test ensures the call doesn't crash and returns the intensity
        result = self.field.execute_substrate_optimization(intensity=1.0)
        self.assertEqual(result, 1.0)

if __name__ == "__main__":
    unittest.main()
