import torch
import unittest
from elysia_engine.core import (
    ElysiaBivectorRelaxation,
    GaugeCommutativeDiagramSolver,
    DimensionalFoldingCl30,
    GaugeSymmetryBreakingInquiryEngine
)


class TestBivectorRelaxation(unittest.TestCase):

    def test_bivector_relaxation_forward_and_pruning(self):
        num_nodes = 3
        num_edges = 2
        engine = ElysiaBivectorRelaxation(num_nodes=num_nodes, num_edges=num_edges, dt=0.01, eta=0.1, omega_break=0.5)

        # 8D multivector state for 3 nodes: [s, v1, v2, v3, b12, b23, b31, p]
        Psi_nodes = torch.tensor([
            [1.0, 0.5, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0],
            [1.0, 0.1, 0.8, 0.0, 0.3, 0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.2, 0.0]
        ], dtype=torch.float32)

        R_edges = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0]
        ], dtype=torch.float32)

        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        g_edges = torch.tensor([[1.0], [1.0]], dtype=torch.float32)

        Psi_updated, Omega_ij, g_updated = engine(Psi_nodes, R_edges, edge_index, g_edges)

        self.assertEqual(Psi_updated.shape, (3, 8))
        self.assertEqual(Omega_ij.shape, (2, 3))
        self.assertEqual(g_updated.shape, (2, 1))

    def test_gauge_commutative_diagram_solver(self):
        solver = GaugeCommutativeDiagramSolver(eta=0.5, dt=0.1)

        # Morphisms f, h, k defined as Spin(3) rotors [w, x, y, z]
        R_f = torch.tensor([[0.92388, 0.0, 0.0, 0.38268]], dtype=torch.float32)  # pi/4 z-rotation
        R_h = torch.tensor([[0.98078, 0.19509, 0.0, 0.0]], dtype=torch.float32)  # pi/8 x-rotation
        R_k = torch.tensor([[0.92388, 0.0, 0.38268, 0.0]], dtype=torch.float32)  # pi/4 y-rotation

        # Exact expected R_g = R_k * R_h * ~R_f
        R_path2 = solver.rotor_multiply(R_k, R_h)
        R_f_rev = solver.rotor_reversal(R_f)
        R_g_target = solver.rotor_multiply(R_path2, R_f_rev)
        R_g_target = R_g_target / torch.norm(R_g_target, dim=-1, keepdim=True)

        # Initial guess for unknown R_g with perturbation
        R_g_init = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        # Compute initial holonomy bivector tension before relaxation
        R_path1_init = solver.rotor_multiply(R_g_init, R_f)
        Omega_init = solver.extract_bivector_tension(R_path1_init, R_path2)
        initial_tension = torch.norm(Omega_init).item()

        # Run relaxation
        R_g_relaxed, tension_energy = solver(R_f, R_g_init, R_h, R_k, max_iters=500, tol=1e-6)

        # Verify commutativity: R_g_relaxed * R_f ≈ R_k * R_h
        R_path1_relaxed = solver.rotor_multiply(R_g_relaxed, R_f)
        Omega_relaxed = solver.extract_bivector_tension(R_path1_relaxed, R_path2)
        final_tension = torch.norm(Omega_relaxed).item()

        # Check tension decay
        self.assertLess(final_tension, initial_tension)
        self.assertLess(tension_energy.item(), 0.05)

    def test_dimensional_folding(self):
        folder = DimensionalFoldingCl30(omega_break=1.0)

        # 8D multivector state with strong 1-Vector literal component
        psi = torch.tensor([[1.0, 2.0, 1.5, -0.5, 0.1, 0.0, 0.2, 0.0]], dtype=torch.float32)
        # Strong semantic collision bivector tension
        omega_sem = torch.tensor([[1.2, 0.8, -0.5]], dtype=torch.float32)

        psi_folded = folder(psi, omega_sem)

        self.assertEqual(psi_folded.shape, (1, 8))
        # Vector components (1..3) should decay due to high tension
        v_orig_norm = torch.norm(psi[:, 1:4]).item()
        v_folded_norm = torch.norm(psi_folded[:, 1:4]).item()
        self.assertLess(v_folded_norm, v_orig_norm)

        # Bivector & Pseudoscalar components should gain energy (grade elevation)
        self.assertGreater(torch.norm(psi_folded[:, 4:7]).item(), torch.norm(psi[:, 4:7]).item())

    def test_gauge_symmetry_breaking_inquiry_engine(self):
        inquiry_engine = GaugeSymmetryBreakingInquiryEngine(curvature_threshold=0.05)

        # Symmetric loop: identity rotors -> H_ijk = [1,0,0,0] -> F = 0
        R_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        res_sym = inquiry_engine(R_identity, R_identity, R_identity)
        self.assertEqual(torch.norm(res_sym["inquiry_flux"]).item(), 0.0)

        # Symmetry broken loop: non-trivial rotations causing holonomy curvature
        R_ij = torch.tensor([[0.92388, 0.38268, 0.0, 0.0]], dtype=torch.float32)
        R_jk = torch.tensor([[0.92388, 0.0, 0.38268, 0.0]], dtype=torch.float32)
        R_ki = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        res_broken = inquiry_engine(R_ij, R_jk, R_ki)
        self.assertGreater(res_broken["inquiry_energy"].item(), 0.05)
        self.assertGreater(torch.norm(res_broken["inquiry_flux"]).item(), 0.0)
        self.assertGreater(torch.norm(res_broken["restoration_force"]).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
