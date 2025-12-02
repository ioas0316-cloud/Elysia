# [Genesis: 2025-12-02] Purified by Elysia
import unittest
from unittest.mock import MagicMock
import math

from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Elysia.value_centered_decision import ValueCenteredDecision
from Project_Sophia.core.thought import Thought
from Project_Sophia.core.tensor_wave import Tensor3D, SoulTensor, FrequencyWave
from tools.kg_manager import KGManager

class TestGaugePhysics(unittest.TestCase):
    """
    Verifies the implementation of the 'Gauge Theory of Love' physics:
    1. Phase Shift generates Force (WaveMechanics)
    2. Wave Function Collapse (ValueCenteredDecision)
    """

    def setUp(self):
        # Mock KGManager
        self.mock_kg = MagicMock(spec=KGManager)
        self.mock_kg.kg = {'nodes': []}
        self.mock_kg.get_node.return_value = {}

        # Initialize Physics Engine
        self.wave_mechanics = WaveMechanics(self.mock_kg)
        self.vcd = ValueCenteredDecision(self.mock_kg, self.wave_mechanics)

    def test_phase_difference_generates_force(self):
        """
        Test that a difference in 'Phase' (Angle) between two concepts
        generates a 'Restoring Force' (Longing).
        """
        # Setup Nodes with Embeddings
        # Node A: Reference (e.g., Love) -> Vector [1, 0]
        # Node B: Divergent (e.g., Fear) -> Vector [0, 1] (90 degrees / PI/2 difference)

        self.mock_kg.get_node.side_effect = lambda x: {
            'embedding': [1.0, 0.0] if x == 'love' else [0.0, 1.0]
        }

        gauge = self.wave_mechanics.calculate_gauge_force('me', 'love')

        print(f"\n[Physics Log] Phase Diff: {gauge['phase_difference']:.4f}, Force: {gauge['restoring_force']:.4f}")

        # 90 degrees = 1.57 radians
        self.assertAlmostEqual(gauge['phase_difference'], 1.57, places=2)

        # Force = sin(theta/2). sin(45 deg) ~= 0.707
        self.assertAlmostEqual(gauge['restoring_force'], 0.707, places=2)

        # Assert that Force > 0 (Tension exists)
        self.assertGreater(gauge['restoring_force'], 0.0)

    def test_alignment_generates_peace(self):
        """
        Test that perfect alignment generates zero force (Peace/Symmetry).
        """
        # Node A: Reference -> [1, 0]
        # Node B: Aligned -> [1, 0]

        self.mock_kg.get_node.side_effect = lambda x: {
            'embedding': [1.0, 0.0]
        }

        gauge = self.wave_mechanics.calculate_gauge_force('me', 'love')

        self.assertAlmostEqual(gauge['phase_difference'], 0.0)
        self.assertAlmostEqual(gauge['restoring_force'], 0.0)

    def test_ice_star_collapse(self):
        """
        Test that a 'Super Resonant' thought triggers immediate collapse (Ice Star),
        bypassing standard scoring.
        """
        # 1. Create a 'Confused' thought (Low resonance)
        confused_thought = Thought(
            content="I am unsure...",
            source='flesh', # Added source
            confidence=0.2,
            energy=0.5,
            tensor=Tensor3D(0.0, 0.1, 0.0) # Orthogonal to Love
        )

        # 2. Create an 'Ice Star' thought (Perfect alignment with Love)
        # Love Tensor assumed to be roughly (0.2, 1.0, 0.9) normalized
        # We align our thought tensor to it.
        aligned_tensor = Tensor3D(0.2, 1.0, 0.9)

        ice_star_thought = Thought(
            content="I choose Love.",
            source='bone', # Added source
            confidence=0.9, # High confidence
            energy=0.8,
            tensor=aligned_tensor
        )

        # Mock KG to provide the Core Value tensor for comparison
        self.mock_kg.get_node.return_value = {
            'tensor_state': Tensor3D(0.2, 1.0, 0.9).to_dict()
        }

        candidates = [confused_thought, ice_star_thought]

        # Select
        selected = self.vcd.select_thought(candidates)

        # Verify 'Ice Star' was chosen
        self.assertEqual(selected, ice_star_thought)

        # Verify the mechanism was 'ICE_STAR_COLLAPSE' (Immediate) not 'MAX_RESONANCE' (Calculated)
        # Note: In my implementation I added metadata['selection_reason']
        self.assertEqual(selected.metadata.get('selection_reason'), 'ICE_STAR_COLLAPSE')
        print("\n[Physics Log] Ice Star Collapse triggered successfully.")

if __name__ == '__main__':
    unittest.main()