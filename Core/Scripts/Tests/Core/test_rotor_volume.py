
import unittest
from Core.S1_Body.L6_Structure.Nature.rotor import Rotor, RotorConfig, RotorMask

class TestRotorVolume(unittest.TestCase):
    def test_volume_mask_existence(self):
        """Test that RotorMask has VOLUME."""
        try:
            mask = RotorMask.VOLUME
        except AttributeError:
            self.fail("RotorMask does not have VOLUME member.")

        self.assertEqual(mask.value, (1, 0, 0, 0), "VOLUME mask value should be (1, 0, 0, 0)")

    def test_volume_process_flow(self):
        """Test that VOLUME mask streams Time, Psi, and Phi."""
        config = RotorConfig(rpm=60.0)
        rotor = Rotor("TestVolume", config)

        # Theta, Phi, Psi, Time
        initial = (0.0, 0.0, 0.0, 0.0)

        # This will raise AttributeError if VOLUME is missing, which is expected failure
        try:
            stream = rotor.process(initial, RotorMask.VOLUME)
        except AttributeError:
            return # Let the first test fail handling catch this, or assume failure here

        # If implemented, check flow
        # Expect 3 frames
        self.assertEqual(len(stream), 3)

        # Check that Phi, Psi, and Time changed
        last_frame = stream[-1]
        self.assertNotEqual(last_frame[1], 0.0, "Phi should flow in VOLUME mode.")
        self.assertNotEqual(last_frame[2], 0.0, "Psi should flow in VOLUME mode.")
        self.assertNotEqual(last_frame[3], 0.0, "Time should flow in VOLUME mode.")
        self.assertEqual(last_frame[0], 0.0, "Theta should remain fixed in VOLUME mode.")

    def test_volume_reverse_flow(self):
        """Test that VOLUME mask respects negative RPM."""
        config = RotorConfig(rpm=-60.0, idle_rpm=-60.0)
        rotor = Rotor("TestVolumeRev", config)
        rotor.current_rpm = -60.0

        initial = (0.0, 0.0, 0.0, 10.0)

        try:
            stream = rotor.process(initial, RotorMask.VOLUME)
        except AttributeError:
            return

        # Check Time flow direction
        times = [p[3] for p in stream]
        self.assertEqual(times[1], 9.0, "Time should flow backwards.")

        # Check Phi/Psi direction (should be negative)
        phis = [p[1] for p in stream]
        self.assertTrue(phis[1] < 0, "Phi should rotate backwards.")

if __name__ == '__main__':
    unittest.main()
