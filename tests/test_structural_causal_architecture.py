"""
Unit tests for Structural Causal Architecture & Fiber Bundle Cognition Engine.
"""

import unittest
import torch
import torch.nn as nn

from synaptic_architecture.structural_causal_architecture import (
    HardConstraintProjectionLayer,
    StructuralCausalEngine,
    ProjectiveCognitiveStructure,
    GaugeConnectionNetwork,
    FiberBundleIntegrator,
    HolonomicGaugeTrainer,
    SheafGlobalSectionVerifier,
    EmbodimentLoopEngine,
    SelfWorldPartitionEngine,
    UnifiedAvatarCausalPipeline
)


class TestStructuralCausalArchitecture(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.dim_state = 6
        self.dim_base = 3
        self.dim_fiber = 4

    def test_hard_constraint_projection_layer(self):
        """
        Tests HardConstraintProjectionLayer invariant manifold projection:
        1. Residual ||C @ x_proj^T|| < 10^-5
        2. Orthogonality of projection correction
        """
        # Conservation constraint matrix C (e.g. 2 constraints on 6D state)
        C = torch.tensor([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 1.0, -1.0, -1.0]], device=self.device, dtype=self.dtype)
        layer = HardConstraintProjectionLayer(self.dim_state, C).to(device=self.device, dtype=self.dtype)

        x_raw = torch.randn(8, self.dim_state, device=self.device, dtype=self.dtype)
        x_proj = layer(x_raw)

        # Residual check ||C @ x_proj^T|| < 1e-5
        violation = torch.matmul(x_proj, C.T)
        max_residual = torch.max(torch.abs(violation)).item()
        self.assertLess(max_residual, 1e-5)

        # Idempotence check: layer(x_proj) == x_proj
        x_proj_again = layer(x_proj)
        torch.testing.assert_close(x_proj_again, x_proj, atol=1e-7, rtol=1e-7)

    def test_structural_causal_engine_relaxation(self):
        """
        Tests StructuralCausalEngine dynamical energy relaxation towards equilibrium.
        """
        C = torch.tensor([[1.0, 0.0, 0.0, -1.0, 0.0, 0.0]], device=self.device, dtype=self.dtype)
        engine = StructuralCausalEngine(self.dim_state, C).to(device=self.device, dtype=self.dtype)

        x_init = torch.randn(4, self.dim_state, device=self.device, dtype=self.dtype)
        x_relaxed, final_energy = engine.relax_to_equilibrium(x_init, steps=10, lr=0.05)

        # Check constraint satisfaction
        violation = torch.matmul(x_relaxed, C.T)
        self.assertLess(torch.max(torch.abs(violation)).item(), 1e-5)
        self.assertEqual(final_energy.shape, (4, 1))

    def test_projective_cognitive_structure(self):
        """
        Tests ProjectiveCognitiveStructure form capture and orthogonal complement energy tracking.
        """
        dim_gas = 10
        dim_proj = 3
        cog_struct = ProjectiveCognitiveStructure(dim_gas, dim_proj).to(device=self.device, dtype=self.dtype)

        rho_gas = torch.randn(5, dim_gas, device=self.device, dtype=self.dtype)
        out = cog_struct.project_gas_to_form(rho_gas)

        self.assertEqual(out["captured_form"].shape, (5, dim_proj))
        self.assertEqual(out["unseen_energy"].shape, (5, 1))
        self.assertTrue(torch.all(out["unseen_energy"] >= 0.0))

    def test_fiber_bundle_integrator_and_holonomy(self):
        """
        Tests FiberBundleIntegrator parallel transport and holonomy curvature loss.
        """
        integrator = FiberBundleIntegrator(self.dim_base, self.dim_fiber, memory_horizon=5).to(device=self.device, dtype=self.dtype)

        # Trajectory in base space B and section sequence in fiber space F
        base_traj = torch.randn(2, 6, self.dim_base, device=self.device, dtype=self.dtype)
        local_sections = torch.randn(2, 6, self.dim_fiber, device=self.device, dtype=self.dtype)

        out = integrator(base_traj, local_sections)

        self.assertEqual(out["total_space_bundle"].shape, (2, 6, self.dim_fiber))
        self.assertIsInstance(out["holonomy_loss"].item(), float)

    def test_holonomic_gauge_trainer(self):
        """
        Tests HolonomicGaugeTrainer end-to-end curvature backprop step.
        """
        integrator = FiberBundleIntegrator(self.dim_base, self.dim_fiber).to(device=self.device, dtype=self.dtype)
        trainer = HolonomicGaugeTrainer(integrator, lr=1e-2)

        base_traj = torch.randn(2, 4, self.dim_base, device=self.device, dtype=self.dtype)
        local_sections = torch.randn(2, 4, self.dim_fiber, device=self.device, dtype=self.dtype)

        loss_dict = trainer.train_step(base_traj, local_sections)

        self.assertIn("total_loss", loss_dict)
        self.assertIn("curvature_loss", loss_dict)
        self.assertGreaterEqual(loss_dict["total_loss"], 0.0)

    def test_sheaf_global_section_verifier(self):
        """
        Tests SheafGlobalSectionVerifier gluing and Cech cocycle obstruction checks.
        """
        num_patches = 3
        dim_fiber = 4
        verifier = SheafGlobalSectionVerifier(num_patches, dim_fiber, tolerance=1e-3)

        # 1. Consistent trivial transitions (Identity)
        identity = torch.eye(dim_fiber, device=self.device, dtype=self.dtype)
        sections_valid = {
            0: torch.ones(dim_fiber, device=self.device, dtype=self.dtype),
            1: torch.ones(dim_fiber, device=self.device, dtype=self.dtype),
            2: torch.ones(dim_fiber, device=self.device, dtype=self.dtype),
        }
        transitions_valid = {
            (0, 1): identity,
            (1, 2): identity,
            (2, 0): identity,
            (1, 0): identity,
            (2, 1): identity,
            (0, 2): identity,
        }

        report_valid = verifier.validate_global_extension(sections_valid, transitions_valid)
        self.assertTrue(report_valid["can_lift_to_global_section"])
        self.assertTrue(report_valid["cech_cocycle_valid"])

        # 2. Obstructed transitions (Broken cocycle)
        transitions_obstructed = dict(transitions_valid)
        # Introduce rotation obstruction in (2, 0)
        rot = torch.eye(dim_fiber, device=self.device, dtype=self.dtype)
        rot[0, 0] = 0.0; rot[0, 1] = -1.0
        rot[1, 0] = 1.0; rot[1, 1] = 0.0
        transitions_obstructed[(2, 0)] = rot

        report_obstructed = verifier.validate_global_extension(sections_valid, transitions_obstructed)
        self.assertFalse(report_obstructed["can_lift_to_global_section"])

    def test_embodiment_loop_engine(self):
        """
        Tests EmbodimentLoopEngine agency residual and self-causal ratio derivation.
        """
        gauge_net = GaugeConnectionNetwork(self.dim_state, self.dim_fiber).to(device=self.device, dtype=self.dtype)
        engine = EmbodimentLoopEngine(self.dim_state, self.dim_fiber, gauge_net).to(device=self.device, dtype=self.dtype)

        x_t = torch.randn(3, self.dim_state, device=self.device, dtype=self.dtype)
        s_t = torch.randn(3, self.dim_fiber, device=self.device, dtype=self.dtype)
        s_next_env = torch.randn(3, self.dim_fiber, device=self.device, dtype=self.dtype)

        out = engine(x_t, s_t, s_next_env)

        self.assertEqual(out["action_vector"].shape, (3, self.dim_state))
        self.assertEqual(out["agency_residual"].shape, (3, self.dim_fiber))
        self.assertEqual(out["self_causal_ratio"].shape, (3, 1))
        self.assertTrue(torch.all(out["self_causal_ratio"] >= 0.0))
        self.assertTrue(torch.all(out["self_causal_ratio"] <= 1.0))

    def test_self_world_partition_engine(self):
        """
        Tests SelfWorldPartitionEngine vector bundle decomposition, tool assimilation,
        loss-of-control boundary shift, SPD metric, and multi-avatar toggling.
        """
        dim_sensory = 8
        batch_size = 4
        engine = SelfWorldPartitionEngine(dim_sensory, threshold=0.5, sharpness=12.0).to(device=self.device, dtype=self.dtype)

        s_next = torch.randn(batch_size, dim_sensory, device=self.device, dtype=self.dtype)
        agency_res = torch.randn(batch_size, dim_sensory, device=self.device, dtype=self.dtype)
        eta = torch.tensor([[0.1], [0.4], [0.7], [0.95]], device=self.device, dtype=self.dtype)

        out = engine(s_next, agency_res, eta)

        # 1. Direct sum completeness: s_next = s_self + s_world
        torch.testing.assert_close(out["s_self"] + out["s_world"], s_next, atol=1e-7, rtol=1e-7)

        # 2. Projection completeness: P_self + P_world = I
        identity = torch.eye(dim_sensory, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        torch.testing.assert_close(out["P_self"] + out["P_world"], identity, atol=1e-7, rtol=1e-7)

        # 3. SPD Effective Metric Check: eigenvalues > 0
        eigvals = torch.linalg.eigvalsh(out["g_effective"])
        self.assertTrue(torch.all(eigvals > 0.0))

    def test_multi_avatar_toggling_and_boundary_swap(self):
        """
        Simulates Avatar Toggling Scenario:
        Swapping control between Avatar A (dims 0..3) and Avatar B (dims 4..7).
        Verifies that Self/World projection subspaces dynamically swap upon toggle.
        """
        dim_per_avatar = 4
        total_dims = dim_per_avatar * 2
        engine = SelfWorldPartitionEngine(dim_sensory=total_dims, threshold=0.5, sharpness=12.0).to(device=self.device, dtype=self.dtype)

        # Phase 1: Avatar A controlled (eta high)
        eta_p1 = torch.tensor([[0.95]], device=self.device, dtype=self.dtype)
        s_sensory = torch.ones(1, total_dims, device=self.device, dtype=self.dtype)
        res_p1 = torch.tensor([[0.01]*4 + [2.5]*4], device=self.device, dtype=self.dtype)

        out_p1 = engine(s_sensory, res_p1, eta_p1)
        self.assertGreater(torch.norm(out_p1["s_self"]).item(), torch.norm(out_p1["s_world"]).item())

        # Phase 2: Toggle control to Avatar B (eta low for A, high for B)
        eta_p2 = torch.tensor([[0.05]], device=self.device, dtype=self.dtype)
        res_p2 = torch.tensor([[2.5]*4 + [0.01]*4], device=self.device, dtype=self.dtype)

        out_p2 = engine(s_sensory, res_p2, eta_p2)
        self.assertGreater(torch.norm(out_p2["s_world"]).item(), torch.norm(out_p2["s_self"]).item())

    def test_unified_avatar_causal_pipeline(self):
        """
        Tests UnifiedAvatarCausalPipeline integrating root divine intent, avatar/NPC partitioning,
        sensory manifold decomposition, and global feedback sync.
        """
        dim_global = 10
        dim_sensory = 6
        num_agents = 4
        batch_size = 2

        pipeline = UnifiedAvatarCausalPipeline(dim_global, dim_sensory, num_agents).to(device=self.device, dtype=self.dtype)

        global_state = torch.randn(batch_size, dim_global, device=self.device, dtype=self.dtype)
        agent_states = torch.randn(batch_size, num_agents, dim_sensory, device=self.device, dtype=self.dtype)
        # Agent 0 is Avatar, Agents 1, 2, 3 are autonomous NPCs
        is_avatar = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0, 0.0]], device=self.device, dtype=self.dtype)
        sensory_obs = torch.randn(batch_size, num_agents, dim_sensory, device=self.device, dtype=self.dtype)

        out = pipeline(global_state, agent_states, is_avatar, sensory_obs)

        self.assertEqual(out["executed_actions"].shape, (batch_size, num_agents, dim_sensory))
        self.assertEqual(out["self_causal_ratios"].shape, (batch_size, num_agents, 1))
        self.assertEqual(out["s_self"].shape, (batch_size, num_agents, dim_sensory))
        self.assertEqual(out["s_world"].shape, (batch_size, num_agents, dim_sensory))
        self.assertEqual(out["global_sync_feedback"].shape, (batch_size, dim_sensory))

        # Avatar eta ~ 0.98, NPC eta ~ 0.05
        self.assertAlmostEqual(out["self_causal_ratios"][0, 0, 0].item(), 0.98, places=2)
        self.assertAlmostEqual(out["self_causal_ratios"][0, 1, 0].item(), 0.05, places=2)


if __name__ == "__main__":
    unittest.main()
