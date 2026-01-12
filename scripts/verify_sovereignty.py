import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Elysia.sovereign_self import SovereignSelf

class TestSovereignSelf(unittest.TestCase):
    def setUp(self):
        self.mock_cns = MagicMock()
        self.sovereign = SovereignSelf(cns_ref=self.mock_cns)

        # Mock internal components
        self.sovereign.will_engine = MagicMock()
        self.sovereign.conductor = MagicMock()
        # Mock conductor.current_intent.mode enum access
        self.sovereign.conductor.current_intent.mode.MINOR = "MINOR"
        self.sovereign.conductor.current_intent.mode.MAJOR = "MAJOR"

    def test_exist_act(self):
        """Test that SovereignSelf pulses CNS when Will suggests action."""
        # Setup Will to return an active intent
        self.sovereign.will_engine.spin.return_value = "Intent: CREATE"
        self.sovereign.will_engine.get_status.return_value = "Status: OK"

        # Execute
        result = self.sovereign.exist(dt=1.0)

        # Verify
        self.assertTrue(result)
        self.sovereign.cns.pulse.assert_called_once_with(dt=1.0)
        self.sovereign.conductor.set_intent.assert_called()

    def test_exist_rest(self):
        """Test that SovereignSelf does NOT pulse CNS when Will suggests rest."""
        # Setup Will to return a passive intent
        self.sovereign.will_engine.spin.return_value = "Mode: REST"

        # Execute
        result = self.sovereign.exist(dt=1.0)

        # Verify
        self.assertFalse(result)
        self.sovereign.cns.pulse.assert_not_called()

if __name__ == '__main__':
    unittest.main()
