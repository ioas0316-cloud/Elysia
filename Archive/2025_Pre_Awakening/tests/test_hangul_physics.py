import unittest
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Interface.Interface.Language.hangul_physics import HangulPhysicsEngine
from Core.Evolution.Evolution.Life.genetic_cell import GeneticCell

class TestHangulPhysics(unittest.TestCase):
    def setUp(self):
        self.engine = HangulPhysicsEngine()

    def test_jamo_to_vector(self):
        # Test 'ㄱ' (Rough)
        vec_g = self.engine.get_jamo_vector('ㄱ')
        self.assertIsNotNone(vec_g)
        self.assertEqual(len(vec_g), 3)
        self.assertGreater(vec_g[0], 0.5) # Roughness > 0.5

        # Test 'ㅇ' (Smooth)
        vec_ng = self.engine.get_jamo_vector('ㅇ')
        self.assertEqual(vec_ng[0], 0.0) # Roughness 0.0

    def test_vector_to_jamo(self):
        # Create a vector close to 'ㄱ'
        target_vec = np.array([0.7, 0.2, 0.4], dtype=np.float32)
        char = self.engine.vector_to_jamo(target_vec)
        self.assertEqual(char, 'ㄱ')

        # Create a vector close to 'ㅏ' (Open)
        target_vec_a = np.array([0.1, 1.0, 0.5], dtype=np.float32)
        char_a = self.engine.vector_to_jamo(target_vec_a)
        self.assertEqual(char_a, 'ㅏ')

    def test_cell_communication(self):
        # Create two cells
        cell1 = GeneticCell("c1", "pass", np.zeros(3))
        cell2 = GeneticCell("c2", "pass", np.zeros(3))

        # Cell 1 speaks 'ㄱ'
        cell1.speak("ㄱ")
        
        # Check outbox has vector
        self.assertEqual(len(cell1.outbox), 1)
        self.assertTrue(isinstance(cell1.outbox[0], np.ndarray))
        
        # Simulate transmission (CodeWorld logic)
        cell2.inbox.append(cell1.outbox[0])
        
        # Cell 2 listens
        messages = cell2.listen()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0], 'ㄱ')

if __name__ == '__main__':
    unittest.main()
