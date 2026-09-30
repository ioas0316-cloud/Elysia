"""
Unit tests for Holographic Clifford Engine (tests/core/physics/test_holographic_clifford_engine.py)
"""

import unittest
import torch
import torch.nn as nn

from core.physics.holographic_clifford_engine import (
    CliffordAlgebra3D,
    TransformerToCliffordBridge,
    CliffordAttention,
    SymbolEmergenceEngine,
    MERAIsometryRG,
    NonEquilibriumSpacetimeField,
    HolographicCognitiveEngine,
)


class TestHolographicCliffordEngine(unittest.TestCase):

    def setUp(self):
        self.device = 'cpu'
        self.ca = CliffordAlgebra3D(device=self.device)

    def test_clifford_algebra_operations(self):
        # 1. Test Cayley Table Shape
        self.assertEqual(self.ca.cayley.shape, (8, 8, 8))

        # 2. Test Geometric Product scalar with scalar
        s1 = torch.zeros(8)
        s1[0] = 2.0  # scalar 2
        s2 = torch.zeros(8)
        s2[0] = 3.0  # scalar 3

        prod = self.ca.geometric_product(s1, s2)
        self.assertAlmostEqual(prod[0].item(), 6.0)

        # 3. Test Basis e1 * e1 = +1
        e1 = torch.zeros(8)
        e1[1] = 1.0
        prod_e1 = self.ca.geometric_product(e1, e1)
        self.assertAlmostEqual(prod_e1[0].item(), 1.0)

        # 4. Test Basis e1 * e2 = e12
        e2 = torch.zeros(8)
        e2[2] = 1.0
        prod_e12 = self.ca.geometric_product(e1, e2)
        self.assertAlmostEqual(prod_e12[4].item(), 1.0)

        # 5. Test Basis e13 * e2 = -e123
        e13 = torch.zeros(8)
        e13[5] = 1.0
        prod_e13_e2 = self.ca.geometric_product(e13, e2)
        self.assertAlmostEqual(prod_e13_e2[7].item(), -1.0)

        # 6. Test Reversion and Norm
        v = torch.randn(8)
        norm_v = self.ca.norm(v)
        self.assertGreaterEqual(norm_v.item(), 0.0)

    def test_transformer_to_clifford_bridge(self):
        d_model = 64
        bridge = TransformerToCliffordBridge(d_model=d_model, device=self.device)

        x = torch.randn(2, 10, d_model)
        res = bridge(x)

        psi = res["psi_multivector"]
        self.assertEqual(psi.shape, (2, 10, 8))

        # Test fusion with existing field
        existing_field = torch.randn(2, 10, 8)
        res_fused = bridge(x, existing_field)
        self.assertEqual(res_fused["psi_multivector"].shape, (2, 10, 8))
        self.assertIsNotNone(res_fused["phase_coherence"])

    def test_clifford_attention(self):
        num_heads = 4
        attn = CliffordAttention(num_heads=num_heads, device=self.device)

        x = torch.randn(2, 8, num_heads, 8)
        out, weights = attn(x)

        self.assertEqual(out.shape, (2, 8, num_heads, 8))
        self.assertEqual(weights.shape, (2, num_heads, 8, 8))

    def test_symbol_emergence_engine(self):
        engine = SymbolEmergenceEngine(initial_num_tiles=3, threshold=0.1, device=self.device)
        self.assertEqual(engine.num_tiles, 3)

        # Create orthogonal field to force mismatch > threshold
        field = torch.randn(2, 4, 8) * 10.0
        res = engine.check_and_sprout_symbol(field)

        self.assertTrue(res["sprouted"])
        self.assertEqual(res["new_num_tiles"], 4)
        self.assertEqual(res["compatibility_matrix"].shape, (4, 4))

    def test_mera_isometry_rg(self):
        mera = MERAIsometryRG(device=self.device)

        # Test 3D sequence input (Batch, Seq_Len, 8)
        seq_field = torch.randn(2, 16, 8)
        macro_seq = mera(seq_field)
        self.assertEqual(macro_seq.shape, (2, 8, 8))

        # Test 4D grid input (Batch, Height, Width, 8)
        grid_field = torch.randn(2, 16, 16, 8)
        macro_grid = mera(grid_field)
        self.assertEqual(macro_grid.shape, (2, 8, 8, 8))

    def test_non_equilibrium_spacetime_field(self):
        field_dyn = NonEquilibriumSpacetimeField(device=self.device)

        psi = torch.randn(2, 8, 8)
        psi_next = field_dyn.evolve_step(psi)
        self.assertEqual(psi_next.shape, (2, 8, 8))

        wave = field_dyn.emit_causal_wave(psi_next)
        self.assertEqual(wave.shape, (2, 8, 8))

        v_causal = field_dyn.compute_causal_potential(psi)
        self.assertGreaterEqual(v_causal.item(), 0.0)

    def test_holographic_cognitive_engine_full_pipeline(self):
        engine = HolographicCognitiveEngine(d_model=32, num_heads=2, device=self.device)

        x_emb = torch.randn(2, 8, 32)
        out = engine(x_emb)

        self.assertIn("psi_multivector", out)
        self.assertIn("psi_attended", out)
        self.assertIn("macro_field", out)
        self.assertIn("psi_evolved", out)
        self.assertIn("emitted_wave", out)
        self.assertIn("causal_potential", out)

        self.assertEqual(out["psi_multivector"].shape, (2, 8, 8))
        self.assertEqual(out["macro_field"].shape, (2, 4, 8))
        self.assertEqual(out["psi_evolved"].shape, (2, 8, 8))


if __name__ == '__main__':
    unittest.main()
