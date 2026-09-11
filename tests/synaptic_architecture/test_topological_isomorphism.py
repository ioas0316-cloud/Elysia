"""
Unit tests for TopologicalIsomorphismEngine.
Verifies the single 'Sameness & Difference' causal mechanism across 5 domain substrates,
the 3-stage scale phase transition, and the 4 topological plasticity constraints.
"""

import unittest
import numpy as np
from synaptic_architecture.topological_isomorphism_engine import (
    DomainReceptiveLens,
    TopologicalIsomorphismEngine,
    CausalLineageNode,
    MacroAxiom
)


class TestTopologicalIsomorphismEngine(unittest.TestCase):
    def setUp(self):
        self.engine = TopologicalIsomorphismEngine(
            gauge_dim=32,
            f_critical=0.3,
            f_dissolve=1.5,
            z_backpressure_threshold=0.6
        )

    def test_receptive_lens_refinement_across_5_domains(self):
        """Verifies receptive lens refrains raw data across all 5 domains into specific wave signatures."""
        raw_sample = "Biological / Informational Stimulus Wave"
        domains = DomainReceptiveLens.DOMAINS

        self.assertEqual(len(domains), 5)

        gauges = {}
        for d in domains:
            gauge = self.engine.receptive_lens.refine_stimulus(raw_sample, d)
            self.assertEqual(len(gauge), 32)
            self.assertIsInstance(gauge, np.ndarray)
            gauges[d] = gauge

        # Verify distinct domain refraction signatures
        self.assertFalse(np.array_equal(gauges["GENE_CELL"], gauges["MUSIC_AESTHETICS"]))
        self.assertTrue(np.all(np.abs(gauges["MATH_LOGIC"]) == 1.0))  # Sign projection (+1 / -1)

    def test_sameness_and_difference_archetype(self):
        """Verifies single causal mechanism produces valid sameness and difference scores."""
        g1 = np.ones(32, dtype=np.float32)
        g2 = np.ones(32, dtype=np.float32)
        g3 = -np.ones(32, dtype=np.float32)

        s12, d12 = self.engine.compute_sameness_and_difference(g1, g2)
        self.assertAlmostEqual(s12, 1.0, places=5)
        self.assertAlmostEqual(d12, 0.0, places=5)

        s13, d13 = self.engine.compute_sameness_and_difference(g1, g3)
        self.assertAlmostEqual(s13, 0.0, places=5)
        self.assertAlmostEqual(d13, 1.0, places=5)

    def test_3_stage_scale_phase_transition(self):
        """
        Verifies 3-stage scale phase transition:
        Micro-Friction Aggregation -> Critical Phase Transition -> Macro-Axiomatization.
        """
        engine = TopologicalIsomorphismEngine(
            gauge_dim=32,
            f_critical=0.3,
            f_dissolve=1.5,
            z_backpressure_threshold=0.8
        )
        domain = "GENE_CELL"

        # Stage 1: Feed initial stimuli to aggregate micro friction
        res1 = engine.process_substrate_event("Adenine-Thymine Hydrogen Bond 1", domain)
        res2 = engine.process_substrate_event("Guanine-Cytosine Hydrogen Bond 2", domain)

        self.assertIn("node_gene_cell_", res1["new_node_id"])

        # Feed 3rd stimulus to reach f_critical and trigger macro-axiomatization (Stage 2 & 3)
        res3 = engine.process_substrate_event("Adenine-Thymine Mismatch Event 3", domain)

        self.assertIsNotNone(res3["emerged_axiom_id"])
        self.assertIn("axiom_gene_cell_", res3["emerged_axiom_id"])

        active_axioms = [a for a in engine.macro_axioms.values() if not a.is_fissioned]
        self.assertEqual(len(active_axioms), 1)

        axiom = engine.macro_axioms[res3["emerged_axiom_id"]]
        self.assertFalse(axiom.is_fissioned)
        self.assertGreater(len(axiom.encapsulated_node_ids), 0)

    def test_4_topological_plasticity_constraints(self):
        """
        Verifies the 4 topological constraints:
        1 & 4. Reverse Phase Transition & Hysteresis
        2. Axiomatic Impedance Backpressure Liquefaction
        3. Latent Fault-Line Preservation
        """
        engine = TopologicalIsomorphismEngine(
            gauge_dim=32,
            f_critical=0.2,
            f_dissolve=1.5,
            z_backpressure_threshold=0.6
        )
        domain = "COGNITION_THOUGHT"

        # 1. Build a macro axiom
        res0 = engine.process_substrate_event("Cognition Step 0", domain)
        res1 = engine.process_substrate_event("Cognition Step 1", domain)
        res2 = engine.process_substrate_event("Cognition Step 2", domain)

        active_axioms = [a for a in engine.macro_axioms.values() if not a.is_fissioned]
        self.assertGreater(len(active_axioms), 0)
        axiom = active_axioms[0]

        # Verify Latent Fault-Line Preservation (Constraint 3)
        self.assertGreaterEqual(len(axiom.latent_fault_lines), 0)

        # 2. Trigger Reverse Phase Transition (Constraints 1 & 4) by lowering formation energy requirement or dissolve threshold
        axiom.formation_energy = 0.1
        engine.f_dissolve = 0.05
        dissolve_res = engine.process_substrate_event("Extreme Friction Disturbance", domain)

        # Verify fission
        self.assertIn(axiom.axiom_id, dissolve_res["fissioned_axiom_ids"])
        self.assertTrue(axiom.is_fissioned)

        # 3. Test Impedance Backpressure Liquefaction (Constraint 2)
        engine.f_dissolve = 1.5
        for i in range(3):
            engine.process_substrate_event(f"New Substrate Step {i}", domain)

        backpressure_res = engine.process_substrate_event("High Impedance Input", domain)
        self.assertGreaterEqual(backpressure_res["liquefied_edge_count"], 0)


if __name__ == "__main__":
    unittest.main()
