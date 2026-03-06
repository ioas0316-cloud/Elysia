import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Core.Cognition.vocation_gravity_engine import VocationGravityEngine, SovereignVector

class MockManifoldEngine:
    def __init__(self):
        self.pulses_injected = []

    def inject_pulse(self, name, energy):
        self.pulses_injected.append((name, energy))

class TestVocationGravityEngine(unittest.TestCase):
    def setUp(self):
        self.engine = MockManifoldEngine()
        self.log_messages = []

        def mock_logger(msg):
            self.log_messages.append(msg)

        self.gravity_engine = VocationGravityEngine(self.engine, log_callback=mock_logger)

        # Set a deterministic vocation vector for testing
        self.gravity_engine.current_vocation_vector = SovereignVector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_calculate_gravity_vector(self):
        conceptual_field = {
            'concept_aligned': SovereignVector([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'concept_orthogonal': SovereignVector([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'concept_anti_aligned': SovereignVector([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        }

        target_id, max_gravity = self.gravity_engine.calculate_gravity_vector(conceptual_field)
        self.assertEqual(target_id, 'concept_aligned')
        self.assertGreater(max_gravity, 0.8) # Strong positive interference

    def test_apply_vocation_torque(self):
        conceptual_field = {
            'concept_aligned': SovereignVector([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        }

        # Apply torque
        self.gravity_engine.apply_vocation_torque(conceptual_field)

        # Check logs
        self.assertTrue(any("Target Concept Pulled: 'concept_aligned'" in msg for msg in self.log_messages))

        # Check if engine received pulse
        self.assertEqual(len(self.engine.pulses_injected), 1)
        self.assertEqual(self.engine.pulses_injected[0][0], "VocationTorque")

        # Check vocation vector evolution
        self.assertAlmostEqual(self.gravity_engine.current_vocation_vector.data[0], 1.09) # 1.0 + 0.9*0.1

if __name__ == '__main__':
    unittest.main()
