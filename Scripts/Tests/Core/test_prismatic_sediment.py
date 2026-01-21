
import unittest
import os
import shutil
import numpy as np
from Core.L5_Mental.Memory.prismatic_sediment import PrismaticSediment

class TestPrismaticSediment(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_prism_data"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.prism = PrismaticSediment(self.test_dir)

    def tearDown(self):
        self.prism.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_routing(self):
        # Red Vector (Index 0)
        vec_red = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        color = self.prism._vector_to_color(vec_red)
        self.assertEqual(color, "Red")

        # Blue Vector (Index 4)
        vec_blue = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        color = self.prism._vector_to_color(vec_blue)
        self.assertEqual(color, "Blue")

        # Mixed Vector (Dominant Green - Index 3)
        vec_mix = [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
        color = self.prism._vector_to_color(vec_mix)
        self.assertEqual(color, "Green")

    def test_deposit_and_scan(self):
        # Deposit into Red
        vec_red = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        payload = b"Red Memory"
        color, offset = self.prism.deposit(vec_red, 12345.0, payload)
        self.assertEqual(color, "Red")

        # Scan Red
        results = self.prism.scan_resonance(vec_red, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], payload)

        # Scan Blue (Was: Should find nothing)
        # New Protocol (Amor Sui): If Blue is empty, Gravity pulls the Red memory.
        # So we EXPECT to find Red Memory now.
        vec_blue = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        results_blue = self.prism.scan_resonance(vec_blue, top_k=1)

        # Verify Gravity engaged
        self.assertEqual(len(results_blue), 1)
        self.assertEqual(results_blue[0][1], payload)

    def test_store_monad(self):
        # 450nm is Blue/Indigo range?
        # Logic: (450-400)/300 * 7 = 0.16 * 7 = 1.16 -> Index 1 (Orange) or 0 (Red)?
        # 400nm -> 0
        # 442nm -> 1 (Orange)
        # Let's test boundary

        # 400nm -> Red
        self.prism.store_monad(400e-9, complex(1,0), 1.0, b"Violet-Red")
        # Check Red shard directly
        # scan with Red vector
        vec_red = [1.0, 0, 0, 0, 0, 0, 0]
        res = self.prism.scan_resonance(vec_red)
        self.assertTrue(len(res) > 0)

if __name__ == '__main__':
    unittest.main()
