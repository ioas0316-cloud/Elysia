"""
Unit tests for StructuralCouplingAndEpistemicGenesisEngine
=========================================================
Verifies continuous tensor phase dynamics across:
1. Multi-scale structural coupling and downward causation.
2. 5-Stage Epistemic Genesis pipeline.
3. Conceptual Reality Causal Mapping (non-physical concepts like money, ghosts, love exerting real potential force).
4. Problem Structure Restructuring Dynamics (Hamiltonian topology transformation).
5. Epistemic Humility tracking (vector phase angle gap between concept and embodied sensation).
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('L1_causal_field'))

from core.consciousness.structural_coupling_epistemic_genesis import StructuralCouplingAndEpistemicGenesisEngine


class TestStructuralCouplingAndEpistemicGenesis(unittest.TestCase):

    def setUp(self):
        self.engine = StructuralCouplingAndEpistemicGenesisEngine(dimension=64)

    def test_multiscale_structural_coupling_and_downward_causation(self):
        res = self.engine.process_multiscale_coupling(
            value_context="치유와 신체 보호",
            numeric_measurement=100.0,
            medium_type="water"
        )

        self.assertIn("v_numeric_norm", res)
        self.assertIn("v_sensory_norm", res)
        self.assertIn("v_bio_hazard_norm", res)
        self.assertIn("v_cog_norm", res)
        self.assertIn("v_value_alignment", res)
        self.assertIn("downward_causation", res)
        self.assertGreater(res["downward_causation"]["numeric_resolution"], 100.0)

    def test_five_stage_epistemic_genesis(self):
        genesis_res = self.engine.execute_epistemic_genesis(
            unexplained_phenomenon="중세 세균 감염 현상",
            observational_limit="현미경 부재 및 미시 유기체 관측 불가"
        )

        self.assertIn("stage_1_differentiation_delta", genesis_res)
        self.assertIn("stage_2_void_energy", genesis_res)
        self.assertIn("stage_3_causal_prediction_norm", genesis_res)
        self.assertIn("stage_4_action_potency", genesis_res)
        self.assertTrue(genesis_res["stage_5_subspace_updated"])

    def test_conceptual_reality_causality_mapping(self):
        ghost_res = self.engine.map_conceptual_reality_causality(
            concept_name="귀신",
            human_context="어두운 밤 고요한 공포"
        )
        self.assertGreater(ghost_res["causal_power_score"], 0.0)
        self.assertIn("impact_force_norm", ghost_res)
        self.assertIn("causal_cascade", ghost_res)

        love_res = self.engine.map_conceptual_reality_causality(
            concept_name="사랑",
            human_context="자기비움과 타자 수용"
        )
        self.assertGreater(love_res["causal_power_score"], 0.0)

    def test_problem_structure_restructuring(self):
        restructure_res = self.engine.restructure_problem_topology(
            surface_problem="시스템 지연 속도 저하",
            current_components=["단선적_수치_연산", "GPU_파라미터_폭주"],
            relational_couplings={"연산": "무상태성_도그마"}
        )

        self.assertIn("resolution_status", restructure_res)
        self.assertGreaterEqual(
            restructure_res["initial_friction_potential"],
            restructure_res["restructured_friction_potential"]
        )

    def test_epistemic_humility_tracking(self):
        humility_res = self.engine.evaluate_and_update_epistemic_humility(
            new_concept_learned="단백질 접힘 인지 구조",
            has_direct_sensory_embodiment=False
        )

        self.assertIn("epistemic_gap", humility_res)
        self.assertIn("phase_gap_rad", humility_res)
        self.assertEqual(humility_res["humility_status"], "HIGH_HUMILITY_AWARENESS_CONCEPTUAL_NOT_FULLY_EMBODIED")

    def test_integrated_coupling_and_genesis_pipeline(self):
        integrated_res = self.engine.process_integrated_coupling_and_genesis(
            input_phenomenon="100도 열 현상과 귀신의 인과력",
            value_context="인간 존중과 가치 판단",
            numeric_val=100.0,
            is_conceptual_only=True
        )

        self.assertIn("multiscale_coupling", integrated_res)
        self.assertIn("epistemic_genesis", integrated_res)
        self.assertIn("conceptual_causality_map", integrated_res)
        self.assertIn("problem_restructuring", integrated_res)
        self.assertIn("epistemic_humility", integrated_res)


if __name__ == "__main__":
    unittest.main()
