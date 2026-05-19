import unittest
from dataclasses import dataclass
from Core.Action.motor_cortex import MotorCortex

# Mock Dependencies
@dataclass
class MockBioSignal:
    pain_level: float
    is_painful: bool

class MockNervousSystem:
    def __init__(self):
        self.pain = 0.0

    def sense(self):
        return MockBioSignal(
            pain_level=self.pain,
            is_painful=self.pain > 0.5
        )

class TestMotorCortex(unittest.TestCase):
    def setUp(self):
        self.ns = MockNervousSystem()
        self.cortex = MotorCortex(nervous_system=self.ns)
        self.cortex.register_actuator("Hand", pin=18)

    def test_normal_movement(self):
        # Initial State: 90
        actuator = self.cortex.actuators["Hand"]
        self.assertEqual(actuator.current_val, 90.0)

        # Drive Forward
        self.cortex.drive("Hand", 10.0)
        self.assertEqual(actuator.current_val, 100.0) # 90 + 10

        # Drive Backward
        self.cortex.drive("Hand", -20.0)
        self.assertEqual(actuator.current_val, 70.0) # 90 - 20

    def test_pain_reflex(self):
        actuator = self.cortex.actuators["Hand"]

        # 1. Normal Move
        self.cortex.drive("Hand", 10.0)
        self.assertEqual(actuator.current_val, 100.0)

        # 2. Trigger Pain
        self.ns.pain = 0.8 # > 0.6 is painful

        # 3. Attempt Move
        self.cortex.drive("Hand", 50.0)

        # Should NOT move to 140. Should be frozen.
        # Note: If frozen, it holds last position or goes to neutral?
        # Current implementation: `move` is not called if frozen.
        # But `drive` logic: if paralyzed, `actuator.freeze()` is called.
        # `freeze` implementation currently does pass.
        # So value should remain 100.0.
        self.assertEqual(actuator.current_val, 100.0)
        self.assertTrue(self.cortex.is_paralyzed)

    def test_recovery(self):
        self.ns.pain = 0.8
        self.cortex.drive("Hand", 50.0) # Trigger paralysis
        self.assertTrue(self.cortex.is_paralyzed)

        self.ns.pain = 0.0
        self.cortex.drive("Hand", 20.0) # Trigger recovery

        self.assertFalse(self.cortex.is_paralyzed)
        self.assertEqual(self.cortex.actuators["Hand"].current_val, 110.0)

if __name__ == "__main__":
    unittest.main()
