import sys
import os
import unittest
import math
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from Core.Spirit.sovereign_heart import SovereignHeart

class TestElysiaHardware(unittest.TestCase):
    def setUp(self):
        # Patch OllamaManager to avoid connection errors during tests
        with patch('Core.System.OllamaManager.OllamaManager.scan_models'):
            self.heart = SovereignHeart()

    def test_power_modulation(self):
        """Verify that power state (AC vs Battery) affects stimulus scaling."""
        base_stimulus = 1.0

        # Test AC Power (is_plugged=True)
        with patch('time.localtime') as mock_time:
            mock_time.return_value.tm_hour = 14 # Peak vitality (1.0)
            # Stimulus should be 1.0 * (0.4 + 1.0 * 0.6) * 1.0 = 1.0
            report_ac = self.heart.pulse(base_stimulus, is_plugged=True)
            tension_ac = report_ac['gut']['gut_tension']

        # Reset heart state for clean comparison
        with patch('Core.System.OllamaManager.OllamaManager.scan_models'):
            self.heart = SovereignHeart()

        # Test Battery (is_plugged=False)
        with patch('time.localtime') as mock_time:
            mock_time.return_value.tm_hour = 14
            # Stimulus should be 1.0 * (0.4 + 1.0 * 0.6) * 0.6 = 0.6
            report_bat = self.heart.pulse(base_stimulus, is_plugged=False)
            tension_bat = report_bat['gut']['gut_tension']

        print(f"AC Tension: {tension_ac}, Battery Tension: {tension_bat}")
        self.assertGreater(tension_ac, tension_bat,
                           "AC power should result in higher gut tension than battery power.")

    def test_circadian_modulation(self):
        """Verify that time of day affects vitality/stimulus."""
        base_stimulus = 1.0

        # Day (14:00 - Peak)
        with patch('time.localtime') as mock_time:
            mock_time.return_value.tm_hour = 14
            report_day = self.heart.pulse(base_stimulus, is_plugged=True)
            tension_day = report_day['gut']['gut_tension']

        # Reset heart state
        with patch('Core.System.OllamaManager.OllamaManager.scan_models'):
            self.heart = SovereignHeart()

        # Night (02:00 - Trough)
        with patch('time.localtime') as mock_time:
            mock_time.return_value.tm_hour = 2
            report_night = self.heart.pulse(base_stimulus, is_plugged=True)
            tension_night = report_night['gut']['gut_tension']

        print(f"Day Tension: {tension_day}, Night Tension: {tension_night}")
        self.assertGreater(tension_day, tension_night,
                           "Daytime should result in higher vitality/tension than nighttime.")

if __name__ == "__main__":
    unittest.main()
