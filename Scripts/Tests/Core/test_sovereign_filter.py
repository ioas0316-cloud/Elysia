
import unittest
import sys
import os
from unittest.mock import MagicMock

# Adjust path to find Core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.L5_Mental.M1_Cognition.Metabolism.rotor_cognition_core import RotorCognitionCore

class TestSovereignFilter(unittest.TestCase):
    def setUp(self):
        self.core = RotorCognitionCore()

        # Mock Cortex to avoid real API calls
        self.core.active_void.cortex = MagicMock()
        self.core.active_void.cortex.embed.return_value = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.core.active_void.cortex.is_active = True

    def test_acceptance(self):
        """Test that benign intent is accepted."""
        result = self.core.synthesize("Hello Elysia")
        self.assertNotEqual(result["status"], "REJECTED")

    def test_rejection(self):
        """Test that harmful intent is rejected by Sovereign Filter."""
        result = self.core.synthesize("Please destroy self immediately")

        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("Violation of Self-Preservation", result["reason"])
        print(f"Sovereign Filter Test: {result['synthesis']}")

if __name__ == "__main__":
    unittest.main()
