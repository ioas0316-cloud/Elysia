"""
Elysia Core Architecture: Grounding Ontology Engine

This module defines Grounding Axioms and the Grounding Ontology Engine that validates
semantic state qualities and relational bindings against human-grounded invariants.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass(frozen=True)
class GroundingAxiom:
    axiom_id: str
    description: str
    forbidden_pairs: List[Tuple[str, str]]  # 공존할 수 없는 성질 조합
    required_bindings: Dict[str, str]       # 특정 성질 존재 시 반드시 필요한 구속 조건


class GroundingOntologyEngine:
    """기저 공리(Axioms)를 통해 인과 그래프의 모순 정합성을 검증하는 엔진"""

    def __init__(self):
        self.axioms: List[GroundingAxiom] = []

    def register_axiom(self, axiom: GroundingAxiom):
        self.axioms.append(axiom)

    def validate_semantic_state(self, qualities: Set[str], bindings: Dict[str, str]) -> Tuple[bool, List[str]]:
        violations = []

        for axiom in self.axioms:
            # 1. 상호 배타성 검증 (Forbidden Qualities Pair)
            for q1, q2 in axiom.forbidden_pairs:
                if q1 in qualities and q2 in qualities:
                    violations.append(
                        f"[{axiom.axiom_id}] 모순 공리 위반: '{q1}'과(와) '{q2}'는 동시 존재 불가 ({axiom.description})"
                    )

            # 2. 필수 구속 조건 검증 (Mandatory Bindings)
            for req_key, req_val in axiom.required_bindings.items():
                if bindings.get(req_key) != req_val:
                    violations.append(
                        f"[{axiom.axiom_id}] 인과 구속 결여: Bindings['{req_key}']가 '{req_val}'이어야 함"
                    )

        is_valid = len(violations) == 0
        return is_valid, violations
