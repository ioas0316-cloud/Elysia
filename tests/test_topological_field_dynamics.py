"""
Unit tests for Continuous Topological Field Dynamics and Hardware-Software Isomorphism.

Tests:
1. Wave injection, superposition, and self-interference.
2. Topological Relaxation to Minimum Energy state (E_min).
3. Irreversible Substrate Conductance Rewiring (Anti-Statelessness).
4. Isomorphic Integration with M-GRIS sticky ends and Dynamic Hardware Mapping.
"""

import unittest
import numpy as np
from core.physics.topological_field_dynamics import TopologicalWaveField, IsomorphicTopologicalEngine
from core.physics.mgris_engine import MGRISInferenceEngine, MGRISCausalBridge, Polarity
from synaptic_architecture.dynamic_hardware_mapping import DynamicHardwareMap


class TestTopologicalFieldDynamics(unittest.TestCase):

    def test_wave_injection_and_interference(self):
        field1 = TopologicalWaveField(grid_size=128)
        field2 = TopologicalWaveField(grid_size=128)

        pattern_a = 0xAAAAAAAAAAAAAAAA
        pattern_b = 0x5555555555555555

        field1.inject_pattern(pattern_a, amplitude=1.0)
        field2.inject_pattern(pattern_b, amplitude=1.0)

        resonance = field1.compute_resonance_with(field2)
        self.assertIsInstance(resonance, float)

    def test_topological_energy_relaxation(self):
        field = TopologicalWaveField(grid_size=128)
        pattern = 0xFF00FF00FF00FF00
        field.inject_pattern(pattern, amplitude=2.0)

        initial_energy = 0.5 * np.sum(field.wave_amplitude ** 2) + 0.5 * np.sum(field.potential_field ** 2)
        steps, final_energy = field.relax_to_equilibrium(max_steps=200, tolerance=1e-3)

        self.assertGreater(steps, 0)
        self.assertLessEqual(final_energy, initial_energy + 1e-5)

    def test_irreversible_substrate_rewiring(self):
        field = TopologicalWaveField(grid_size=128)
        initial_conductance = np.copy(field.conductance_substrate)

        field.inject_pattern(0x123456789ABCDEF0)
        field.relax_step()

        # Apply feedback energy
        field.apply_irreversible_feedback(feedback_energy=2.5, focal_index=32)

        # Conductance substrate must be permanently altered
        self.assertFalse(np.array_equal(initial_conductance, field.conductance_substrate))
        self.assertGreater(field.conductance_substrate[32], initial_conductance[32])

    def test_isomorphic_engine_cycle(self):
        engine = IsomorphicTopologicalEngine(grid_size=128)
        result = engine.process_isomorphic_cycle(0xCAFEBABEDEADBEEF, feedback=1.5)

        self.assertIn("input_bitmask", result)
        self.assertIn("emergent_bitmask", result)
        self.assertIn("relaxation_steps", result)
        self.assertIn("final_energy", result)
        self.assertGreater(result["substrate_conductance_mean"], 1.0)

    def test_mgris_and_hardware_mapping_integration(self):
        # 1. Test Dynamic Hardware Mapping with Wave Field Refraction
        dhm = DynamicHardwareMap(size=1024)
        bitstream = np.uint64(0x8888888888888888)
        addr1 = dhm.derive_address(bitstream)

        self.assertIsInstance(addr1, int)
        self.assertGreaterEqual(addr1, 0)
        self.assertLess(addr1, 1024)

        # 2. Test M-GRIS 2-Morphism wave field reconfiguration
        mgris = MGRISInferenceEngine()
        node_q = MGRISCausalBridge.create_concept_node(0, "A", "B")
        node_op = MGRISCausalBridge.create_concept_node(1, "NOT", "NOT", is_operator=True)

        graph, narrative = mgris.execute_inference_cycle(
            query_strand=node_q,
            knowledge_pool=[node_op],
            max_depth=2
        )
        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
