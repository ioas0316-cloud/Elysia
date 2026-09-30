import unittest
import torch
import math
from core.memory.geometric_folding_engine import GeometricFoldingEngine
from core.memory.hardware_aware_clifford_pipeline import HardwareAwareCliffordPipeline

class TestGeometricFoldingPipeline(unittest.TestCase):

    def setUp(self):
        self.folding_engine = GeometricFoldingEngine(dim=8)
        self.hardware_pipeline = HardwareAwareCliffordPipeline()

    def test_bivector_wedge(self):
        v1 = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        v2 = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        B = self.folding_engine.compute_bivector_wedge(v1, v2)
        # e1 ^ e2 = e12 (index 4)
        self.assertAlmostEqual(B[0, 4].item(), 1.0, places=5)
        self.assertAlmostEqual(B[0, 5].item(), 0.0, places=5)
        self.assertAlmostEqual(B[0, 6].item(), 0.0, places=5)

    def test_rotor_generation(self):
        Bivector = torch.zeros(1, 8)
        Bivector[0, 4] = 1.0 # e12
        theta = torch.tensor([math.pi / 2.0])

        R = self.folding_engine.generate_rotor(Bivector, theta)
        # Scalar part = cos(pi/4) ~ 0.7071
        # Bivector part = -sin(pi/4) ~ -0.7071
        self.assertAlmostEqual(R[0, 0].item(), math.cos(math.pi / 4.0), places=4)
        self.assertAlmostEqual(R[0, 4].item(), -math.sin(math.pi / 4.0), places=4)

    def test_geometric_folding_forward(self):
        v1 = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        v2 = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        theta = torch.tensor([0.5])

        V_promoted = self.folding_engine(v1, v2, theta)
        self.assertEqual(V_promoted.shape, (1, 8))
        # Grade-2 component at e12 (idx 4) should be 1.0
        self.assertAlmostEqual(V_promoted[0, 4].item(), 1.0, places=5)

    def test_hardware_metric_and_attenuation(self):
        vram_metric = self.hardware_pipeline.compute_hardware_metric('vram')
        ram_metric = self.hardware_pipeline.compute_hardware_metric('ram')
        ssd_metric = self.hardware_pipeline.compute_hardware_metric('ssd')

        self.assertAlmostEqual(vram_metric, 1.0, places=5)
        self.assertTrue(ram_metric > vram_metric)
        self.assertTrue(ssd_metric > ram_metric)

        memory_blocks = {
            'vram': torch.ones(1, 8),
            'ram': torch.ones(1, 8),
            'ssd': torch.ones(1, 8)
        }
        psi_total = self.hardware_pipeline(memory_blocks)
        self.assertEqual(psi_total.shape, (1, 8))

if __name__ == '__main__':
    unittest.main()
