"""
[Test Suite: Meta-Axiomatic Evaluator & Topological Reframing Integration]
1. 블랙박스 외부 공리의 경계 섭동 반응 및 O(1) 목적성 마찰 지표 산출 검증
2. 내부/외부 공리 간 상대적 마찰 대조 (compare_and_decide) 및 CausalReframingEngine 자율 이식 검증
3. 우월 외부 공리 채택 후 StaticBypassManager 연동을 통한 무지성 텐서 연산 0(Zero Bypass) 소멸 검증
4. 섭동 역설 (Perturbation Paradox): 미세 섭동 주입 시 internal 공리는 붕괴하나 external 공리는 안정적인 특성을 가려내는 유용성 입증
"""

import unittest
from typing import Any, List

from synaptic_architecture.meta_axiomatic_evaluator import (
    IntentInvariant,
    ExternalAxiomBlackBox,
    FrictionWeightConfig,
    PurposeFrictionMetrics,
    MetaAxiomaticEvaluator,
)
from synaptic_architecture.causal_reframing_engine import CausalReframingEngine
from synaptic_architecture.topological_axiomatic_engine import TopologicalAxiomaticEngine
from synaptic_architecture.non_tensor_meta_boundary import StaticBypassManager, SymmetryState


class DummyInternalAxiomBlackBox:
    """내부 좁은 공리 체계 (고정된 국소 규칙만 수행, 섭동 시 불안정)"""

    def __init__(self, rigid_limit: float = 10.0):
        self.rigid_limit = rigid_limit

    @property
    def axiom_signature(self) -> str:
        return "rigid_internal_axiom_v1"

    def project_intent(self, intent_input: Any) -> Any:
        val = float(intent_input)
        if val > self.rigid_limit:
            # 좁은 경계를 넘어설 경우 섭동에 의한 출력 붕괴/이상치 발생
            return {"output": -999.0, "valid": False}
        return {"output": val * 1.0, "valid": True}

    def get_structural_complexity_cost(self) -> float:
        # 국소 규칙 유지를 위해 예외 처리가 가중된 오버헤드
        return 0.6


class DummySuperiorExternalAxiomBlackBox:
    """외부 우월 상위 공리 체계 (블랙박스: 내부 구조 미노출, 섭동에도 유연 이완 및 불변성 보존)"""

    def __init__(self):
        pass

    @property
    def axiom_signature(self) -> str:
        return "smooth_external_manifold_axiom_v2"

    def project_intent(self, intent_input: Any) -> Any:
        val = float(intent_input)
        # 높은 차원의 연속적 인과 궤적 처리
        return {"output": val * 1.05, "valid": True}

    def get_structural_complexity_cost(self) -> float:
        # 상위 간결성(MDL 원리)에 따라 낮은 유지 비용
        return 0.1


class TestMetaAxiomaticEvaluator(unittest.TestCase):

    def setUp(self):
        # 최상위 불변 조건 정의: 출력값이 valid=True 이고 output >= 0.0 인 경우
        self.core_invariant = IntentInvariant(
            intent_id="I_core_positive_validity",
            invariant_checker=lambda res: isinstance(res, dict) and res.get("valid", False) and res.get("output", -1) >= 0.0,
            perturbation_tolerance=0.1
        )
        self.evaluator = MetaAxiomaticEvaluator(core_invariant=self.core_invariant)
        self.bypass_mgr = StaticBypassManager()
        self.topo_engine = TopologicalAxiomaticEngine(bypass_manager=self.bypass_mgr)
        self.reframing_engine = CausalReframingEngine(axiomatics_engine=self.topo_engine)

    def test_blackbox_boundary_perturbation_and_friction_measurement(self):
        """
        [1] 외부 공리의 내부 로직을 스캔하지 않고,
        오직 경계 투입/불변성 반응 및 섭동 관찰만으로 O(1) 마찰 지표 산출
        """
        external_axiom = DummySuperiorExternalAxiomBlackBox()
        test_intents = [1.0, 1.1, 1.2, 1.3, 1.4]

        metrics = self.evaluator.observe_friction(external_axiom, test_intents)

        self.assertIsInstance(metrics, PurposeFrictionMetrics)
        self.assertEqual(metrics.invariance_violation_rate, 0.0)
        self.assertEqual(metrics.complexity_overhead, 0.1)
        self.assertLess(metrics.total_friction, 0.3)

    def test_relative_comparison_and_autonomous_reframing(self):
        """
        [2] compare_and_decide()를 통한 내부/외부 공리 대조 및
        CausalReframingEngine을 통한 외부 우월 공리 자율 채택/이식 검증
        """
        internal_axiom = DummyInternalAxiomBlackBox(rigid_limit=5.0)
        external_axiom = DummySuperiorExternalAxiomBlackBox()

        # 5.0 이상의 intent 투입 시 internal axiom은 불변성 파괴 발생
        test_intents = [1.0, 3.0, 6.0, 8.0, 10.0]

        decision = self.evaluator.compare_and_decide(
            internal_axiom=internal_axiom,
            external_axiom=external_axiom,
            sample_intents=test_intents
        )

        self.assertTrue(decision["adopt_external"])
        self.assertLess(decision["external_friction"], decision["internal_friction"])

        # CausalReframingEngine과의 연동
        result = self.reframing_engine.evaluate_and_reframe_axioms(
            internal_axiom=internal_axiom,
            external_axiom=external_axiom,
            evaluator=self.evaluator,
            sample_intents=test_intents,
            target_domain="Universal_Physics_Boundary"
        )

        self.assertTrue(result["reframed"])
        self.assertIsNotNone(result["reframed_signature_id"])
        self.assertTrue(result["zero_bypass_achieved"])

    def test_zero_computation_bypass_after_axiom_rebind(self):
        """
        [3] 외부 우월 공리가 채택되어 TopologicalAxiomaticEngine에 Re-bind 된 후,
        평형 상태에서 StaticBypassManager를 통해 하부 텐서 연산이 0으로 완전 소멸(Zero Bypass)되는지 입증
        """
        internal_axiom = DummyInternalAxiomBlackBox(rigid_limit=2.0)
        external_axiom = DummySuperiorExternalAxiomBlackBox()
        test_intents = [1.0, 5.0]

        reframing_res = self.reframing_engine.evaluate_and_reframe_axioms(
            internal_axiom=internal_axiom,
            external_axiom=external_axiom,
            evaluator=self.evaluator,
            sample_intents=test_intents,
            target_domain="Hardware_VRAM_Scheduler"
        )

        sig_id = reframing_res["reframed_signature_id"]

        # Re-bound 된 정적 타입 구속하에서 O(1) Zero Computation Bypass 수행
        proof, is_bypassed, cb_res = self.topo_engine.resolve_with_zero_bypass(
            signature_id=sig_id,
            current_transition=("Meta_Input_Boundary", "Meta_Response_State"),
            active_tension=0.0,
            i_meta_boundary_balanced=True,
            tensor_callback=lambda: "HEAVY_GPU_COMPUTATION_EXECUTED"
        )

        self.assertTrue(is_bypassed)
        self.assertIsNone(cb_res)  # 하부 Heavy GPU 연산은 단 1회도 실행되지 않고 0으로 정적 소멸됨
        self.assertEqual(proof.symmetry_state, SymmetryState.PRESERVED)

    def test_perturbation_paradox_scenario(self):
        """
        [4] 섭동 역설 (Perturbation Paradox) 검증:
        평시(작은 intent)에는 두 공리가 모두 정답을 도출하는 것으로 보이나,
        미세 섭동(micro perturbation)을 가했을 때 internal 공리는 급격한 불연속/붕괴(High Instability)를 일으키는 반면,
        external 공리는 우아하게 이완(Low Instability)됨을 MetaAxiomaticEvaluator가 정확히 판별해내는지 입증
        """
        internal_rigid = DummyInternalAxiomBlackBox(rigid_limit=10.0)
        external_smooth = DummySuperiorExternalAxiomBlackBox()

        # 평시 sample intents: [1.0, 2.0, 3.0] -> 둘 다 불변성 통과
        normal_intents = [1.0, 2.0, 3.0]
        internal_norm_m = self.evaluator.observe_friction(internal_rigid, normal_intents)
        external_norm_m = self.evaluator.observe_friction(external_smooth, normal_intents)

        # 평시에는 불변성 파괴가 모두 0
        self.assertEqual(internal_norm_m.invariance_violation_rate, 0.0)
        self.assertEqual(external_norm_m.invariance_violation_rate, 0.0)

        # 미세 섭동 및 경계 영역 주입: [9.0, 10.5, 12.0, 15.0]
        perturbed_intents = [9.0, 10.5, 12.0, 15.0]

        decision = self.evaluator.compare_and_decide(
            internal_axiom=internal_rigid,
            external_axiom=external_smooth,
            sample_intents=perturbed_intents
        )

        # 섭동 주입 시 internal 공리의 마찰 지표가 급증하여 Evaluator가 외부 공리를 채택함
        self.assertTrue(decision["adopt_external"])
        self.assertGreater(decision["internal_metrics"].invariance_violation_rate, 0.0)
        self.assertEqual(decision["external_metrics"].invariance_violation_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
