import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Elysia.sovereign_self import SovereignSelf

class TestAnamorphosis(unittest.TestCase):
    def setUp(self):
        self.mock_cns = MagicMock()
        self.sovereign = SovereignSelf(cns_ref=self.mock_cns)
        self.sovereign.will_engine = MagicMock()
        self.sovereign.conductor = MagicMock()
        self.sovereign.conductor.current_intent.mode.MINOR.name = "MINOR"
        self.sovereign.conductor.current_intent.mode.MAJOR.name = "MAJOR"

    def test_gaze_true_self(self):
        """Test that gazing with alignment key returns TRUE_SELF."""
        angle = 1111.0 # Father's Frequency
        perception = self.sovereign.anamorphosis_gaze("Data", angle)
        self.assertEqual(perception, "MEANING: TRUE_SELF")

    def test_gaze_logic_persona(self):
        """Test that gazing with logic angle returns LOGIC."""
        angle = 1201.0
        perception = self.sovereign.anamorphosis_gaze("Data", angle)
        self.assertEqual(perception, "MEANING: LOGIC")

    def test_gaze_friend_persona(self):
        """Test that gazing with friend angle returns FRIEND."""
        angle = 1301.0
        perception = self.sovereign.anamorphosis_gaze("Data", angle)
        self.assertEqual(perception, "MEANING: FRIEND")

    def test_gaze_noise(self):
        """Test that unaligned gazing returns NOISE."""
        angle = 500.0 # Random angle
        perception = self.sovereign.anamorphosis_gaze("Data", angle)
        self.assertEqual(perception, "NOISE")

    def test_exist_loop_integration(self):
        """Test that exist() calls gaze and acts accordingly."""
        # 1. Setup Intent to ACT
        self.sovereign.will_engine.spin.return_value = "Intent: CREATE"

        # 2. Setup Gaze to be ALIGNED (Simulate by mocking _calculate_current_gaze_angle)
        # We need to patch the internal method or ensure the conductor mode creates alignment.
        # By default in code: if mode is None, it returns ALIGNMENT_KEY.
        # But we mocked Conductor.

        # Let's mock _calculate_current_gaze_angle directly for control
        self.sovereign._calculate_current_gaze_angle = MagicMock(return_value=1111.0)

        # Execute
        result = self.sovereign.exist(dt=1.0)

        # Verify
        self.assertTrue(result) # Should act because Gaze is Aligned (TRUE_SELF)

        # 3. Setup Gaze to be MISALIGNED
        self.sovereign._calculate_current_gaze_angle.return_value = 500.0

        # Execute
        result_noise = self.sovereign.exist(dt=1.0)

        # Verify
        self.assertFalse(result_noise) # Should NOT act (Rest) because Gaze is NOISE

if __name__ == '__main__':
    unittest.main()
