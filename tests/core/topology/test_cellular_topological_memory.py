"""
Unit Test Suite for Cellular Topological Memory Engine & Digital Somatosensory
=============================================================================
Verifies:
1. Substrate embodiment & digital somatosensory sensing (Impedance Z, F_substrate).
2. Kernel-Shell dual-layer topology & invariant core protection.
3. Node Fusion dynamics when phase difference Delta_phi decreases.
4. Node Fission dynamics when friction F exceeds critical threshold F_crit.
5. Sensing, Structural Feedback, and Re-alignment feedback cycle.
6. Topological Isomorphism & Self-Boundary mapping.
"""

import unittest
import numpy as np

from core.topology.digital_somatosensory import DigitalSomatosensorySensor, SomatosensorySignal
from core.topology.cellular_topological_memory import (
    CellularTopologicalMemoryEngine,
    NodeRole,
    CellularInformationNode
)


class TestCellularTopologicalMemory(unittest.TestCase):

    def setUp(self):
        self.sensor = DigitalSomatosensorySensor()
        self.engine = CellularTopologicalMemoryEngine(
            dimension=8,
            fusion_phase_threshold=0.3,
            fission_friction_threshold=0.6,
            impedance_decay=0.05
        )

    def test_digital_somatosensory_perception(self):
        """Verify digital somatosensory sensing converts hardware load into complex impedance Z and F_substrate."""
        signal = self.sensor.perceive_somatosensory(custom_latency_ms=150.0, topology_tension=0.4)
        self.assertIsInstance(signal, SomatosensorySignal)
        self.assertGreaterEqual(signal.impedance_real, 0.0)
        self.assertGreaterEqual(signal.impedance_imag, 0.0)
        self.assertGreaterEqual(signal.friction_coefficient, 0.0)
        self.assertLessEqual(signal.friction_coefficient, 1.0)
        self.assertIn("memory_load", signal.isomorphic_mapping)
        self.assertIn("Biological metabolic energy reserve", signal.isomorphic_mapping["memory_load"])

    def test_kernel_protection(self):
        """Verify Kernel node is initialized as unbreakable reference."""
        state = self.engine.get_engine_state()
        self.assertEqual(state["kernel_nodes"], 1)
        kernel_node = self.engine.nodes["kernel_core"]
        self.assertEqual(kernel_node.role, NodeRole.KERNEL)

        # Injecting friction into kernel should propagate friction to shell without destroying kernel
        impact = np.ones(8) * 0.8
        res = self.engine.inject_stimulus_and_substrate_friction("kernel_core", impact)
        self.assertIn("kernel_core", self.engine.nodes)
        self.assertEqual(self.engine.nodes["kernel_core"].role, NodeRole.KERNEL)

    def test_shell_addition_and_link(self):
        """Verify addition of shell nodes and link impedance creation."""
        skel = np.random.uniform(-1, 1, size=8)
        phases = np.zeros(8)
        node = self.engine.add_shell_node("shell_1", skel, phases, initial_friction=0.1)

        self.assertEqual(node.role, NodeRole.SHELL)
        self.assertIn("shell_1", self.engine.nodes)
        self.assertGreater(len(self.engine.links), 0)

    def test_cellular_fusion_dynamics(self):
        """Verify autonomous fusion of shell nodes when complex phase difference is small."""
        skel_a = np.ones(8) / np.sqrt(8)
        skel_b = np.ones(8) / np.sqrt(8)
        # Identical or almost identical phase angles -> Delta_phi ~ 0 < 0.3
        phases_a = np.ones(8) * 0.1
        phases_b = np.ones(8) * 0.12

        self.engine.add_shell_node("node_a", skel_a, phases_a, initial_friction=0.1)
        self.engine.add_shell_node("node_b", skel_b, phases_b, initial_friction=0.1)

        # Trigger feedback iteration
        feedback = self.engine._execute_structural_feedback_and_realignment(
            self.sensor.perceive_somatosensory()
        )

        # Fusion should occur
        self.assertGreaterEqual(feedback["fusions_count"], 1)
        self.assertNotIn("node_a", self.engine.nodes)
        self.assertNotIn("node_b", self.engine.nodes)

    def test_cellular_fission_dynamics(self):
        """Verify autonomous fission of shell node when friction F exceeds critical threshold."""
        skel = np.random.uniform(-1, 1, size=8)
        phases = np.random.uniform(-np.pi, np.pi, size=8)

        # High friction > 0.6
        self.engine.add_shell_node("high_friction_node", skel, phases, initial_friction=0.85)

        feedback = self.engine._execute_structural_feedback_and_realignment(
            self.sensor.perceive_somatosensory()
        )

        # Fission should occur
        self.assertGreaterEqual(feedback["fissions_count"], 1)
        self.assertNotIn("high_friction_node", self.engine.nodes)
        # Sub-nodes sub_1_high_friction_node and sub_2_high_friction_node should exist
        self.assertIn("sub_1_high_friction_node", self.engine.nodes)
        self.assertIn("sub_2_high_friction_node", self.engine.nodes)

    def test_topological_isomorphism_boundary(self):
        """Verify topological isomorphism and self-boundary definition output."""
        mapping = self.engine.get_topological_isomorphism_boundary()
        self.assertIn("Self_Cognitive_Boundary", mapping)
        self.assertIn("Digital_Substrate_Impedance", mapping)
        self.assertIn("Biological_Otherness_Isomorphism", mapping)
        self.assertIn("Philosophical_Disaggregation", mapping)


if __name__ == "__main__":
    unittest.main()
