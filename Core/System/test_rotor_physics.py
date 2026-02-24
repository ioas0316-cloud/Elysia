
import unittest
from Core.System.rotor import Rotor, RotorConfig, RotorMask

class TestRotorPhysics(unittest.TestCase):
    def test_reverse_spin(self):
        """Test that negative RPM causes the angle to decrease."""
        config = RotorConfig(rpm=-60.0, idle_rpm=-60.0)
        rotor = Rotor("TestReverse", config)

        # Force current RPM to negative for immediate testing
        rotor.current_rpm = -60.0
        rotor.target_rpm = -60.0

        initial_angle = rotor.current_angle

        # Update for 1 second. At -60 RPM, it should rotate -360 degrees.
        # But since we mod 360, let's do 0.5 seconds -> -180 degrees.
        rotor.update(0.5)

        # Calculate expected angle
        expected_angle = (initial_angle - 180.0) % 360.0

        print(f"Initial: {initial_angle}, After 0.5s (-60RPM): {rotor.current_angle}, Expected: {expected_angle}")

        # Allow small float error
        self.assertAlmostEqual(rotor.current_angle, expected_angle, delta=0.1,
                               msg="Rotor angle did not decrease correctly with negative RPM.")

    def test_time_reversal_stream(self):
        """Test that negative RPM causes 'process' to stream time backwards."""
        config = RotorConfig(rpm=-60.0, idle_rpm=-60.0)
        rotor = Rotor("TestTime", config)
        rotor.current_rpm = -60.0

        # Theta, Phi, Psi, Time
        coords = (0.0, 0.0, 0.0, 10.0)

        # Mask LINE means Time Flows.
        # Currently it adds +1.0 per step. We expect -1.0 if RPM is negative.
        stream = rotor.process(coords, RotorMask.LINE)

        # We expect 3 frames.
        # Frame 0: Time 10.0
        # Frame 1: Time 9.0
        # Frame 2: Time 8.0

        times = [p[3] for p in stream]
        print(f"Stream Times with Negative RPM: {times}")

        self.assertEqual(times[1], 9.0, "Time did not flow backwards in the stream.")
        self.assertEqual(times[2], 8.0, "Time did not flow backwards in the stream.")

    def test_axis_existence(self):
        """Test that Rotor has an axis property."""
        config = RotorConfig(rpm=60.0)
        rotor = Rotor("TestAxis", config)

        # Check if attribute exists
        self.assertTrue(hasattr(rotor, 'axis'), "Rotor is missing 'axis' attribute.")
        self.assertTrue(hasattr(config, 'axis'), "RotorConfig is missing 'axis' attribute.")

if __name__ == '__main__':
    unittest.main()
