
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from Core.Memory.Mind.resonance_engine import ResonanceEngine

class MockStorage:
    def get_all_concepts(self):
        # Yield mock concepts
        # [id, [wx,wy,wz], emotions, values, subs, tokens, m_count, m_int, cat, lat, ac, q]
        # Quantization: 255=1.0, 127=0.0, 0=-1.0
        
        # Concept A: (1.0, 0.0, 0.0) -> [255, 127, 127]
        yield "A", ["A", [255, 127, 127], {}, {}, [], [], 0, 0, 0, 0, 0, 0]
        
        # Concept B: (1.0, 0.0, 0.0) -> [255, 127, 127] (Similar to A)
        yield "B", ["B", [255, 127, 127], {}, {}, [], [], 0, 0, 0, 0, 0, 0]
        
        # Concept C: (0.0, 1.0, 0.0) -> [127, 255, 127] (Orthogonal)
        yield "C", ["C", [127, 255, 127], {}, {}, [], [], 0, 0, 0, 0, 0, 0]
        
        # Concept D: (-1.0, 0.0, 0.0) -> [0, 127, 127] (Opposite)
        yield "D", ["D", [0, 127, 127], {}, {}, [], [], 0, 0, 0, 0, 0, 0]

class TestResonanceEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ResonanceEngine()
        self.storage = MockStorage()
        self.engine.build_index(self.storage)
        
    def test_build_index(self):
        self.assertEqual(len(self.engine.ids), 4)
        self.assertIn("A", self.engine.id_to_idx)
        
    def test_resonance(self):
        # Query with A's vector (1, 0, 0)
        # Should find B (similar), then C (orthogonal), then D (opposite)
        # Note: D might be last or excluded depending on threshold, but here we just check order
        
        results = self.engine.find_resonance([1.0, 0.0, 0.0], k=3, exclude_id="A")
        
        # B should be first (score ~1.0)
        self.assertEqual(results[0][0], "B")
        self.assertAlmostEqual(results[0][1], 1.0, places=1)
        
        # C should be second (score ~0.0)
        self.assertEqual(results[1][0], "C")
        self.assertAlmostEqual(results[1][1], 0.0, places=1)
        
    def test_add_vector(self):
        self.engine.add_vector("E", [0.5, 0.5, 0.0])
        self.assertIn("E", self.engine.id_to_idx)
        
        # Query E
        results = self.engine.find_resonance([0.5, 0.5, 0.0], k=1, exclude_id="E")
        # Should find A or C as they are 45 degrees away
        self.assertTrue(results[0][0] in ["A", "B", "C"])

if __name__ == "__main__":
    unittest.main()
