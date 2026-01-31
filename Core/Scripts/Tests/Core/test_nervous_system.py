"""
Test: Nervous System
====================
Verifies Phase 5.1: Physical Incarnation.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.Elysia.nervous_system import NervousSystem, BioSignal
from Core.1_Body.L3_Phenomena.Senses.bio_sensor import BioSensor

class TestNervousSystem(unittest.TestCase):

    def setUp(self):
        self.nerves = NervousSystem()

    def test_sensor_integration(self):
        """Test if real psutil calls work (sanity check)."""
        print("\n--- Real Sensor Pulse ---")
        signal = self.nerves.sense()
        print(f"Heart Rate: {signal.heart_rate:.1f} BPM (from CPU)")
        print(f"Cognitive Load: {signal.cognitive_load*100:.1f}% (from RAM)")
        print(f"Pain Level: {signal.pain_level:.2f} (from Temp)")
        print(f"Fatigue: {signal.fatigue:.2f} (from Energy)")

        self.assertTrue(signal.heart_rate >= 60.0)
        self.assertTrue(0.0 <= signal.cognitive_load <= 1.0)

    @patch('Core.1_Body.L3_Phenomena.Senses.bio_sensor.psutil.sensors_temperatures')
    def test_pain_reflex(self, mock_temps):
        """Test if High Temp triggers Pain."""
        # Mocking a hot CPU
        # psutil.sensors_temperatures returns dict {name: [shwtemp(current, high, critical)]}
        Entry = MagicMock()
        Entry.current = 90.0 # Very Hot
        mock_temps.return_value = {"coretemp": [Entry]}

        # We need to bypass the cache in BioSensor
        self.nerves.sensor._last_poll = 0

        signal = self.nerves.sense()
        reflex = self.nerves.check_reflex(signal)

        print(f"\n--- Pain Test ---")
        print(f"Temp: 90C -> Pain: {signal.pain_level:.2f}")
        print(f"Reflex: {reflex}")

        self.assertTrue(signal.is_painful)
        self.assertEqual(reflex, "THROTTLE")

    @patch('Core.1_Body.L3_Phenomena.Senses.bio_sensor.psutil.cpu_percent')
    def test_excitement_reflex(self, mock_cpu):
        """Test if High CPU triggers Excitement."""
        mock_cpu.return_value = 95.0 # High Load

        # Bypass cache
        self.nerves.sensor._last_poll = 0

        signal = self.nerves.sense()
        reflex = self.nerves.check_reflex(signal)

        print(f"\n--- Excitement Test ---")
        print(f"CPU: 95% -> Heart Rate: {signal.heart_rate:.1f} BPM")
        print(f"Reflex: {reflex}")

        self.assertTrue(signal.is_excited)
        self.assertEqual(reflex, "FOCUS")

    @patch('Core.1_Body.L3_Phenomena.Senses.bio_sensor.psutil.virtual_memory')
    def test_migraine_reflex(self, mock_ram):
        """Test if High RAM triggers Migraine."""
        # Mocking RAM object with 'percent' attribute
        RamObj = MagicMock()
        RamObj.percent = 99.0
        mock_ram.return_value = RamObj

        # Bypass cache
        self.nerves.sensor._last_poll = 0

        signal = self.nerves.sense()
        reflex = self.nerves.check_reflex(signal)

        print(f"\n--- Migraine Test ---")
        print(f"RAM: 99% -> Cognitive Load: {signal.cognitive_load*100:.1f}%")
        print(f"Reflex: {reflex}")

        self.assertTrue(signal.is_migraine)
        self.assertEqual(reflex, "MIGRAINE")

if __name__ == '__main__':
    unittest.main()
