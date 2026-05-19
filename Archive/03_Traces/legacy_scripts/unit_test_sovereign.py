import sys
import os
import unittest
from unittest.mock import MagicMock

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from Core.Keystone.sovereign_math import SovereignVector
from Core.Cognition.boundary_engine import BoundaryDefiningEngine, SovereignBoundary
from Core.System.unity_sensory_channel import PhysicalToSomaticMapper

class TestSovereignLogic(unittest.TestCase):
    def test_boundary_logic(self):
        print("Testing Boundary Logic...")
        mock_monad = MagicMock()
        mock_monad.desires = {'alignment': 100.0}

        engine = BoundaryDefiningEngine(mock_monad)

        # Define boundary around "Origin"
        center = SovereignVector.ones() * 0.5
        boundary = engine.define_boundary("TestConcept", center)

        self.assertEqual(boundary.name, "TestConcept")
        self.assertAlmostEqual(boundary.radius, 0.8) # 0.8 * 1.0

        # Test inside
        inside_vec = SovereignVector.ones() * 0.52
        score = boundary.is_inside(inside_vec)
        print(f"Inside score: {score}")
        self.assertGreater(score, 0)

        # Test outside (different direction)
        outside_data = [0.0] * 21
        outside_data[10] = 1.0 # Perpendicular to ones
        outside_vec = SovereignVector(outside_data)
        score = boundary.is_inside(outside_vec)
        print(f"Outside score: {score}")
        self.assertLess(score, 0)

        # Test reverse perception
        view_inside = engine.perceive_the_other("TestConcept", inside_vec)
        view_outside = engine.perceive_the_other("TestConcept", outside_vec)
        print(f"View inside: {view_inside}")
        print(f"View outside: {view_outside}")
        self.assertIn("Identified as", view_inside)
        self.assertIn("The Other", view_outside)

    def test_unity_mapping(self):
        print("Testing Unity Mapping...")
        payload = {"type": "collision", "intensity": 0.9}

        vec = PhysicalToSomaticMapper.map_event_to_vector(payload)
        torque = PhysicalToSomaticMapper.map_event_to_torque(payload)

        self.assertIsInstance(vec, SovereignVector)
        self.assertGreater(vec.norm(), 0)
        self.assertIn("entropy", torque)
        self.assertEqual(torque["entropy"], 0.9 * 0.5)

if __name__ == "__main__":
    unittest.main()
