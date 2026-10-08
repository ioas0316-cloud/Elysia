"""
Unit and Integration Tests for Dynamic Execution Field Architecture
===================================================================
Tests JIT dynamic compilation, LRU cache 0ms recall, multi-scale quaternion rotor execution,
perceptual-motor resonance & wonder evaluation, spontaneous JIT code synthesis,
autopoietic entropy tracking & pruning, and biographical narrative reflection.
"""

import time
import unittest
import numpy as np
import torch

from synaptic_architecture.jit_synapse_bridge import (
    DynamicJITBridge,
    InMemoLRUKernelCache,
    rotate_vector_quaternion_np
)
from synaptic_architecture.perceptual_motor_curiosity import (
    PerceptualMotorResonanceEngine,
    SpontaneousCodeSynthesizer
)
from synaptic_architecture.narrative_autopoiesis import (
    SelfhoodAutopoiesisEngine,
    BiographicalEpiphanyMoment
)
from synaptic_architecture.dynamic_execution_field import (
    AutopoieticTensorODE,
    DynamicExecutionField
)


class TestDynamicExecutionFieldArchitecture(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.jit_bridge = DynamicJITBridge(force_cpu=True)

    def test_lru_cache_and_zero_latency_recall(self):
        cache = InMemoLRUKernelCache(capacity=2)
        cache.put("KEY1", "HANDLE1")
        cache.put("KEY2", "HANDLE2")

        self.assertEqual(cache.get("KEY1"), "HANDLE1")
        cache.put("KEY3", "HANDLE3")  # Evicts KEY2

        self.assertEqual(cache.get("KEY1"), "HANDLE1")
        self.assertEqual(cache.get("KEY3"), "HANDLE3")
        self.assertIsNone(cache.get("KEY2"))

    def test_multiscale_quaternion_rotor_execution(self):
        in_data = torch.randn(100, 3)
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.7071, 0.0, 0.7071, 0.0]])
        weights = torch.tensor([1.0, 0.5])

        out1, friction1 = self.jit_bridge.execute_multiscale_rotor(in_data, quats, weights, "TEST_ROTOR_KEY")
        self.assertFalse(friction1["cache_hit"])
        self.assertEqual(out1.shape, (100, 3))

        # Re-execution (Cache hit)
        out2, friction2 = self.jit_bridge.execute_multiscale_rotor(in_data, quats, weights, "TEST_ROTOR_KEY")
        self.assertTrue(friction2["cache_hit"])
        self.assertEqual(friction2["compile_time_ms"], 0.0)
        torch.testing.assert_close(out1, out2)

    def test_perceptual_motor_resonance_and_wonder(self):
        engine = PerceptualMotorResonanceEngine(feature_dim=64)
        hypothesis = torch.sin(torch.linspace(0, 3.14, 64))
        world_aligned = torch.sin(torch.linspace(0, 3.14, 64))
        world_perturbed = torch.sin(torch.linspace(0, 3.14, 64)) + 0.8 * torch.cos(torch.linspace(0, 3.14, 64))

        p_res1, wonder1, diss1 = engine.evaluate_resonance_and_wonder(hypothesis, world_aligned)
        p_res2, wonder2, diss2 = engine.evaluate_resonance_and_wonder(hypothesis, world_perturbed)

        self.assertGreater(p_res1, p_res2)
        self.assertGreater(wonder2, wonder1)
        self.assertGreater(diss2, diss1)

    def test_spontaneous_code_synthesis_and_execution(self):
        synthesizer = SpontaneousCodeSynthesizer()
        cpp_code, hash_key = synthesizer.generate_harmonic_cpp_code(wonder_index=0.75, alpha=0.35)

        test_data = np.linspace(0, 10, 100, dtype=np.float32)
        out, friction = self.jit_bridge.execute_custom_harmonic_kernel(
            cpp_code, test_data, alpha=0.35, concept_hash_key=hash_key
        )

        self.assertEqual(out.shape, (100,))
        self.assertIn("execution_latency_ms", friction)

    def test_autopoietic_entropy_and_pruning(self):
        auto = SelfhoodAutopoiesisEngine(max_entropy_threshold=0.5)

        # Small dissonance
        entropy1, threat1 = auto.evaluate_structural_entropy(dissonance=0.2)
        self.assertFalse(threat1)

        # Critical dissonance
        entropy2, threat2 = auto.evaluate_structural_entropy(dissonance=1.5)
        self.assertTrue(threat2)

        pruned_msg = auto.execute_autopoietic_pruning()
        self.assertIn("PRUNED", pruned_msg)
        self.assertLess(auto.structural_entropy, 0.5)

    def test_narrative_epiphany_and_reflection(self):
        auto = SelfhoodAutopoiesisEngine()
        auto.record_epiphany_moment(p_resonance=0.92, wonder_index=0.65, kernel_tag="ResonantRotorV1")

        self.assertEqual(len(auto.biography_moments), 1)
        reflection = auto.reflect_first_person_narrative()
        self.assertIn("epiphany", reflection)
        self.assertIn("0.920", reflection)

    def test_autopoietic_tensor_ode_limit_cycle(self):
        ode = AutopoieticTensorODE(num_scales=3, latent_dim=16, dt=0.01)
        Z = torch.complex(torch.randn(1, 3, 16), torch.randn(1, 3, 16))
        g = torch.ones(1, 3, 16)
        world = torch.complex(torch.randn(1, 3, 16) * 0.01, torch.randn(1, 3, 16) * 0.01)

        for _ in range(10):
            Z, g = ode.step_rk4(Z, g, world)

        self.assertEqual(Z.shape, (1, 3, 16))
        self.assertEqual(g.shape, (1, 3, 16))
        self.assertFalse(torch.isnan(Z).any())
        self.assertFalse(torch.isnan(g).any())

    def test_dynamic_execution_field_unified_loop(self):
        field = DynamicExecutionField(num_scales=3, latent_dim=32, force_cpu_jit=True)
        Z = torch.complex(torch.randn(1, 3, 32) * 0.1, torch.randn(1, 3, 32) * 0.1)
        g = torch.ones(1, 3, 32)
        world_stream = torch.randn(1, 32)

        # Cycle 1
        res1 = field.process_field_cycle(Z, g, world_stream, wonder_threshold=0.01)
        self.assertTrue(res1["spontaneous_code_generated"])
        self.assertIsNotNone(res1["jit_friction"])
        self.assertIn("epiphany", res1["narrative_reflection"])


if __name__ == "__main__":
    unittest.main()
