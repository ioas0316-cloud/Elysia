"""
Unit Tests for Elysia Core Topology: Neural-to-Phase Interface & Dynamic Elastic Engine
=======================================================================================
"""

import unittest
import torch
import math

from core.topology.neural_phase_interface import (
    NeuralToPhaseInterface,
    IntegratedCognitivePipeline,
    DifferentiableNeuralToPhaseInterface,
    DifferentiableCausalManifoldBuilder,
    LatentGradientFeedbackRefiner,
    HybridCausalInferencePipeline,
    ElasticProfile,
    VRAMSensors,
    ElasticLatentGenerator,
    MicroLatentGenerator,
    ElasticCausalBuilder,
    DynamicElasticEngine
)


class TestNeuralPhaseInterface(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        torch.manual_seed(42)

    def test_neural_to_phase_interface(self):
        latent_dim = 128
        interface = NeuralToPhaseInterface(latent_dim=latent_dim, context_dim=4, device=self.device)

        dummy_z = torch.randn(1, latent_dim, device=self.device)
        out = interface(dummy_z)

        self.assertIn("theta", out)
        self.assertIn("context_plane", out)
        self.assertIn("focus_delta", out)

        self.assertTrue(0 <= out["theta"] <= 2 * math.pi)
        self.assertEqual(out["context_plane"].shape, (4, 4))
        self.assertGreaterEqual(out["focus_delta"], 0.0)

    def test_integrated_cognitive_pipeline(self):
        pipeline = IntegratedCognitivePipeline(latent_dim=128)
        dummy_z = torch.randn(1, 128)
        dummy_phoneme = torch.randn(16)

        result = pipeline.step_cognitive_cycle("node_test_01", dummy_z, dummy_phoneme)

        self.assertIn("mapped_theta_deg", result)
        self.assertIn("focus_delta", result)
        self.assertIn("det", result)
        self.assertIn("is_contradiction", result)
        self.assertEqual(result["final_causal_matrix"].shape, (4, 4))

    def test_latent_gradient_feedback_refiner(self):
        interface = DifferentiableNeuralToPhaseInterface(latent_dim=128, context_dim=4)
        builder = DifferentiableCausalManifoldBuilder(dim=4)
        refiner = LatentGradientFeedbackRefiner(interface, builder)

        z_init = torch.randn(1, 128)
        base_phoneme = torch.randn(16)

        res = refiner.refine_latent_intent(
            z_init=z_init,
            base_phoneme=base_phoneme,
            max_iters=10,
            lr=0.05,
            det_threshold=1e-3,
            lambda_least_action=0.01
        )

        self.assertIn("z_refined", res)
        self.assertIn("final_det", res)
        self.assertIn("iterations_taken", res)
        self.assertTrue(res["z_refined"].shape == z_init.shape)

    def test_hybrid_causal_inference_pipeline(self):
        interface = DifferentiableNeuralToPhaseInterface(latent_dim=128, context_dim=4)
        builder = DifferentiableCausalManifoldBuilder(dim=4)
        pipeline = HybridCausalInferencePipeline(
            transformer_model=None,
            phase_interface=interface,
            causal_builder=builder,
            threshold=1.0  # Force hallucination threshold high to test masking logic
        )

        dummy_h_t = torch.randn(1, 128)
        base_phoneme = torch.randn(16)

        res = pipeline.step_filter_inference(
            input_ids=torch.tensor([[1, 2]]),
            base_phoneme=base_phoneme,
            override_h_t=dummy_h_t
        )

        self.assertIn("logits", res)
        self.assertIn("det_val", res)
        self.assertIn("is_hallucination", res)
        self.assertTrue(res["is_hallucination"])

    def test_dynamic_elastic_engine_profiles(self):
        # Test Micro Profile
        micro_prof = ElasticProfile("Micro Mode Test", 128, 4, 64, 3, 1e-3)
        micro_engine = DynamicElasticEngine(override_profile=micro_prof, in_features=64)

        dummy_input = torch.randn(1, 32, 64)  # [Batch, Seq_Len, Features]
        out_micro = micro_engine.process_sequence(dummy_input)

        self.assertEqual(out_micro["latent_z"].shape, (1, 128))
        self.assertEqual(out_micro["manifold_m"].shape, (1, 4, 4))

        # Test Standard Profile
        std_prof = ElasticProfile("Standard Mode Test", 512, 8, 256, 6, 1e-4)
        std_engine = DynamicElasticEngine(override_profile=std_prof, in_features=64)
        out_std = std_engine.process_sequence(dummy_input)

        self.assertEqual(out_std["latent_z"].shape, (1, 512))
        self.assertEqual(out_std["manifold_m"].shape, (1, 8, 8))

        # Test Expanded Profile
        exp_prof = ElasticProfile("Expanded Mode Test", 2048, 16, 512, 12, 1e-5)
        exp_engine = DynamicElasticEngine(override_profile=exp_prof, in_features=64)
        out_exp = exp_engine.process_sequence(dummy_input)

        self.assertEqual(out_exp["latent_z"].shape, (1, 2048))
        self.assertEqual(out_exp["manifold_m"].shape, (1, 16, 16))

    def test_vram_sensors(self):
        profile = VRAMSensors.detect_and_get_profile()
        self.assertIsNotNone(profile)
        self.assertIn(profile.latent_dim, [128, 512, 2048])


if __name__ == "__main__":
    unittest.main()
