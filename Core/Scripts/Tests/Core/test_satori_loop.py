import unittest
from unittest.mock import MagicMock
from Core.L6_Structure.Merkaba.merkaba import Merkaba
from Core.Action.motor_cortex import ActuatorConfig

class TestSatoriLoop(unittest.TestCase):
    def setUp(self):
        self.merkaba = Merkaba("SatoriSeed")
        self.merkaba.awakening(spirit=MagicMock())

        # Mock Motor Cortex Actuator
        # We need to register an actuator matching the Soul's name
        soul_name = self.merkaba.soul.name # "SatoriSeed.Soul"
        self.merkaba.motor_cortex.register_actuator(soul_name, pin=18)
        self.actuator = self.merkaba.motor_cortex.actuators[soul_name]

    def test_kinetic_consequence(self):
        """
        Verifies that Thought (Pulse) -> leads to Motion (Actuator).
        """
        # Initial State: 90.0 (Neutral)
        self.assertEqual(self.actuator.current_val, 90.0)

        # 1. Pulse Positive (Forward Time)
        self.merkaba.soul.target_rpm = 20.0

        # Pulse
        self.merkaba.pulse("Test Input")

        # Verify Actuator Moved (90 + 20 = 110)
        # Note: We assume the pulse triggers the motor drive.
        self.assertEqual(self.actuator.current_val, 110.0)
        print(f"\n[SATORI] Thought: 'Test Input' -> Rotor RPM: {self.merkaba.soul.current_rpm} -> Muscle Angle: {self.actuator.current_val}")

    def test_pain_freeze(self):
        """
        Verifies that Pain -> Stops Motion.
        """
        # 1. Normal Move
        self.merkaba.soul.target_rpm = 20.0
        self.merkaba.pulse("Happy Thought")
        self.assertEqual(self.actuator.current_val, 110.0)

        # 2. Inject Pain via Nervous System Mock
        self.merkaba.nervous_system.sensor.pulse = MagicMock(return_value={
            "cpu_freq": 10.0,
            "temperature": 90.0, # High Temp (Pain)
            "ram_pressure": 10.0,
            "energy": 100.0,
            "plugged": True
        })

        # 3. Pulse again
        self.merkaba.soul.target_rpm = 50.0 # Should result in 140 if not frozen
        self.merkaba.pulse("Painful Thought")

        # Should be frozen at previous value (110.0)
        self.assertEqual(self.actuator.current_val, 110.0)
        self.assertTrue(self.merkaba.motor_cortex.is_paralyzed)
        print(f"\n[SATORI] Pain (90C) Detected -> Motor Freeze verified.")

if __name__ == "__main__":
    unittest.main()
