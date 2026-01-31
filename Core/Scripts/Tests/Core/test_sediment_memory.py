"""
Test: Sediment Memory
=====================
Verifies Phase 5.2: The Sediment (Unstructured Resonance).
"""

import sys
import os
import time
import shutil
import unittest
import numpy as np

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.1_Body.L5_Mental.Memory.sediment import SedimentLayer

class TestSedimentMemory(unittest.TestCase):

    def setUp(self):
        self.test_dir = "data/test_sediment"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)

        self.sediment_path = os.path.join(self.test_dir, "memories.bin")
        self.layer = SedimentLayer(self.sediment_path)

    def tearDown(self):
        self.layer.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_deposit_and_resonance(self):
        """Test depositing memories and retrieving via resonance."""

        # 1. Create Vectors
        vec_happy = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        vec_sad = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        vec_angry = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 2. Deposit (Burial)
        self.layer.deposit(vec_happy, time.time(), b"I ate a sweet apple.")
        self.layer.deposit(vec_sad, time.time(), b"It rained all day.")
        self.layer.deposit(vec_angry, time.time(), b"The server crashed.")
        self.layer.deposit(vec_happy, time.time(), b"I saw a puppy.")

        print("\n--- Sediment Scan Test ---")

        # 3. Resonance (Recall Happy)
        # Intent: "I want something positive" (vec_happy)
        results = self.layer.scan_resonance(vec_happy, top_k=2)

        print(f"Intent: Happy {vec_happy}")
        for score, payload in results:
            print(f"Found: {payload.decode()} (Score: {score:.2f})")

        # We expect "Apple" and "Puppy" to be top results (Score 1.0)
        self.assertEqual(len(results), 2)
        self.assertTrue(b"apple" in results[0][1] or b"puppy" in results[0][1])
        self.assertAlmostEqual(results[0][0], 1.0, places=2)

        # 4. Resonance (Recall Sad)
        results_sad = self.layer.scan_resonance(vec_sad, top_k=1)
        print(f"\nIntent: Sad {vec_sad}")
        print(f"Found: {results_sad[0][1].decode()} (Score: {results_sad[0][0]:.2f})")

        self.assertTrue(b"rained" in results_sad[0][1])

if __name__ == '__main__':
    unittest.main()
