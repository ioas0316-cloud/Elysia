"""
Elysia Core Architecture: Dual Plasticity Pipeline

This module links bottom-up sensory VQ projection with top-down grounding ontology
verification and AST rewriting, contracting sensory activation thresholds (tau)
when ontology violations occur to maintain homeostatic stability.
"""

from typing import Any, Dict, List, Set, Tuple
from core.embodied.projector import VQSensoryProjector
from core.topology.grounding_ontology import GroundingOntologyEngine
from core.topology.ast_rewriting import SelfRewritingNodeEngine


class DualPlasticityPipeline:
    """하위 감각 가소성(VQ Threshold)과 상위 연산 가소성(AST Rewriting)을 동시에 구동하는 파이프라인"""

    def __init__(self, vq_projector: VQSensoryProjector, ontology: GroundingOntologyEngine = None, rewriter: SelfRewritingNodeEngine = None):
        self.vq_projector = vq_projector
        self.ontology = ontology or GroundingOntologyEngine()
        self.rewriter = rewriter or SelfRewritingNodeEngine()

    def process_sensory_input(self, raw_sensory_vector: List[float], initial_bindings: Dict[str, str] = None) -> Dict[str, Any]:
        # Step 1: Bottom-up 감각 투영 (VQ Quantization)
        qualities, distances = self.vq_projector.project_to_qualities(raw_sensory_vector)
        print(f"📡 [Bottom-up Sensing] 투영된 Qualities: {qualities} (거리: {distances})")

        bindings = initial_bindings or {"SAFETY_CONTAINMENT": "INACTIVE"}

        # Step 2: Top-down 온톨로지 공리 검증
        is_valid, violations = self.ontology.validate_semantic_state(qualities, bindings)

        # Step 3: 공리 위반 시 2중 피드백(Dual Plasticity) 발동
        if not is_valid:
            print(f"\n🚨 [Ontology Violation] 상충 기호 또는 인과 구속 결여 감지! Top-down 역방향 피드백 발동")
            for v in violations:
                print(f"  ├─ {v}")

            # 1) 하위 피드백: 충돌을 유발한 감각 프로젝터의 발화 임계값 수축 (임계값 억제)
            # 모순이 발생한 symbol 탐색
            for symbol_name in list(qualities):
                if any(symbol_name in v for v in violations):
                    self.vq_projector.adapt_threshold(symbol_name, contraction_factor=0.5)

            # 2) 재투영 실행: 수축된 임계값 기반으로 감각 신호 재평가
            corrected_qualities, _ = self.vq_projector.project_to_qualities(raw_sensory_vector)
            print(f"✨ [Re-Projected] 억제 적용 후 재투영된 Qualities: {corrected_qualities}")
            return {"qualities": corrected_qualities, "bindings": {"SAFETY_CONTAINMENT": "ACTIVE"}, "violations": violations}

        return {"qualities": qualities, "bindings": bindings, "violations": []}
