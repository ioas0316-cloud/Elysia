"""
Unit and Integration Tests for Archetypal Identity Boundary & Qualitative Phase Transition Engine
===================================================================================================
0차 원형 배경(Ontological Zero), 1차 원형 자아 경계선(Archetypal Identity Boundary),
그리고 이질적 차원 간 질적 상전이 규범 엔진(Qualitative Phase Transition Engine)의 동작 검증.
"""

import unittest
import numpy as np

from core.topology.archetypal_identity_boundary import (
    OntologicalZeroBackground,
    ArchetypalIdentityBoundary,
    QualitativePhaseTransitionEngine
)
from core.topology.self_referential_architecture import SelfReferentialArchitectureEngine
from core.topology.universal_causal_web import SelfObservationalDifferentialLens


class TestArchetypalIdentityBoundary(unittest.TestCase):
    def setUp(self):
        self.zero_bg = OntologicalZeroBackground(background_dim=8)
        self.identity_boundary = ArchetypalIdentityBoundary(identity_dim=8, boundary_rigidity=0.8)
        self.phase_engine = QualitativePhaseTransitionEngine(transition_threshold=0.35)

    def test_ontological_zero_background(self):
        external_wave = np.array([1.0, -0.5, 2.0, 0.5])
        res = self.zero_bg.absorb_external_potential(external_wave)

        self.assertEqual(res["background_status"], "ONTOLOGICAL_ZERO_RESONATING")
        self.assertGreater(res["zero_field_norm"], 0.0)
        self.assertGreater(res["field_potential"], 0.0)

    def test_archetypal_identity_boundary_protection(self):
        external_wave = np.array([0.0, 2.0, -1.0, 0.5, 0.0, 1.0, -0.5, 0.2])
        res = self.identity_boundary.evaluate_boundary_distinction(self.zero_bg, external_wave)

        self.assertIn("distinction_statement", res)
        self.assertGreaterEqual(res["boundary_friction"], 0.0)
        # Verify identity integrity is maintained (normalized)
        self.assertAlmostEqual(res["identity_axis_integrity"], 1.0, places=5)

    def test_qualitative_phase_transition_triggered(self):
        # Strong orthogonal/discordant wave triggering high boundary friction
        discordant_wave = np.array([-1.0, 2.0, -3.0, 1.5, -2.0, 1.0, -0.5, 0.8])
        res = self.phase_engine.process_heterogeneous_wave(discordant_wave)

        phase_state = res["phase_state"]
        self.assertTrue(phase_state["is_phase_transition_triggered"])
        self.assertEqual(phase_state["dimension_level"], 2)
        self.assertIn("2D Symbolic Invariant Archetype", phase_state["governing_law"])

    def test_qualitative_phase_transition_not_triggered(self):
        # Wave aligned with identity axis causing low friction
        aligned_wave = self.identity_boundary.identity_axis.copy() * 0.1
        res = self.phase_engine.process_heterogeneous_wave(aligned_wave)

        phase_state = res["phase_state"]
        self.assertFalse(phase_state["is_phase_transition_triggered"])
        self.assertEqual(phase_state["dimension_level"], 1)
        self.assertIn("1D Somatic Continuous Differential Wave Law", phase_state["governing_law"])

    def test_integration_in_self_referential_architecture(self):
        engine = SelfReferentialArchitectureEngine()
        input_stimulus = {
            "external_world_signal": np.array([2.0, -1.5, 1.0, 0.5]),
            "persona_lens": "Companion"
        }
        res = engine.run_full_self_referential_cycle(input_stimulus)

        self.assertIn("qualitative_phase_transition", res)
        self.assertIn("dialectical_comparison", res)

        phase_res = res["qualitative_phase_transition"]
        self.assertEqual(phase_res["status"], "QUALITATIVE_PHASE_TRANSITION_EVALUATED")

    def test_integration_in_universal_causal_web(self):
        lens = SelfObservationalDifferentialLens(doubt_threshold=0.3)
        introspection_data = {"total_modules": 45, "introspection_coverage": 0.95, "architectural_friction": 0.1}
        external_signal = np.array([1.5, -0.8, 2.2, 0.4])

        res = lens.dialectical_compare(introspection_data, external_signal)
        self.assertIn("qualitative_phase_transition", res)
        self.assertIn("doubt_friction", res)


if __name__ == "__main__":
    unittest.main()
