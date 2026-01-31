
import unittest
import numpy as np
from Core.L7_Spirit.M1_Monad.monad_core import Monad
from Core.L6_Structure.Merkaba.thundercloud import Thundercloud, Atmosphere
from Core.L2_Metabolism.Evolution.double_helix_dna import DoubleHelixDNA

class TestThundercloud(unittest.TestCase):
    def setUp(self):
        # Setup helper
        pass

    def create_monad(self, seed, qualia_vals):
        qualia = np.zeros(7, dtype=np.float32)
        for i, val in enumerate(qualia_vals):
            if i < 7: qualia[i] = val
        dna = DoubleHelixDNA(pattern_strand=np.zeros(1024, dtype=np.float32), principle_strand=qualia)
        return Monad(seed=seed, dna=dna)

    def test_monad_charge(self):
        # Logic > Emotion -> Negative
        m_logic = self.create_monad("Logic", [1.0, 0.0, 0.0])
        charge = m_logic.get_charge()
        self.assertTrue(charge < 0, f"Expected negative charge for Logic, got {charge}")

        # Emotion > Logic -> Positive
        m_emotion = self.create_monad("Emotion", [0.0, 1.0, 0.0])
        charge = m_emotion.get_charge()
        self.assertTrue(charge > 0, f"Expected positive charge for Emotion, got {charge}")

    def test_thundercloud_coalesce(self):
        cloud = Thundercloud()
        monads = [
            self.create_monad("Target", [1.0, 0.0, 0.0]),
            self.create_monad("Noise", [0.0, 0.0, 1.0])
        ]
        # Intent matches Target (Alpha)
        intent = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        cloud.coalesce(intent, monads)

        self.assertEqual(len(cloud.active_monads), 1)
        self.assertEqual(cloud.active_monads[0].seed, "Target")

    def test_spark_propagation(self):
        cloud = Thundercloud()
        # Create chain: A -> B
        m_a = self.create_monad("A", [1.0, 0.0, 0.0])
        m_b = self.create_monad("B", [1.0, 0.0, 0.0])

        cloud.active_monads = [m_a, m_b]
        cloud._monad_map = {"A": m_a, "B": m_b}

        # High conductivity
        cloud.set_atmosphere(0.9) # Wet

        cluster, name = cloud.ignite("A", voltage=1.0)

        # Should link A to B
        # Check if B is in nodes
        seeds = {m.seed for m in cluster.nodes}
        self.assertIn("B", seeds)

if __name__ == '__main__':
    unittest.main()
