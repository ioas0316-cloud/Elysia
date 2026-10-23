"""
Linguistic Cognition Engine (언어적 사고 및 동형성 인과 엔진)
=============================================================================
수치 계산이나 통계 모델링 없이, 언어적 기호의 상징적 인과망(Symbolic Causal Network)과
동형성(Isomorphism)에 기반하여 진단하고 의사결정을 내리는 언어적 사고 엔진.

4대 기호적 인과 기전:
1. 상징적 인과 집속 (Symbolic Causal Compression)
2. 은유와 위상적 전이 (Metaphorical Transfer)
3. 맥락적 중력장과 자율 억제/촉발 (Semantic Gravity Field)
4. 서사적 결합성 (Narrative Coherence)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set


@dataclass
class CausalRelation:
    """언어적 기호 간의 맥락적 인과 관계"""
    cause_symbol: str
    effect_symbol: str
    relation_type: str       # 예: "IMPEDES_FLOW", "ATTEMPTS_REMEDY", "AMPLIFIES_PRESSURE", "ELIMINATES_OBSTRUCTION"
    narrative_coherence: float # 서사적 개연성 (0.0 ~ 1.0)
    description: str = ""


@dataclass
class CausalSymbolNode:
    """언어적 기호 노드 (수치로 분해되지 않는 독립 인과 상징)"""
    symbol: str
    domain: str              # 예: "BIOLOGICAL", "ORGANIZATIONAL", "ENVIRONMENTAL"
    compressed_causes: List[str] = field(default_factory=list)
    compressed_effects: List[str] = field(default_factory=list)
    relational_ties: List[CausalRelation] = field(default_factory=list)


class LinguisticCognitionEngine:
    """수학적 환원 없이 작동하는 언어적 사고(Linguistic Cognition) 엔진"""

    def __init__(self):
        self.symbol_network: Dict[str, CausalSymbolNode] = {}
        self.isomorphic_mappings: List[Dict[str, Any]] = []
        self._initialize_baseline_symbolic_causal_graphs()

    def _initialize_baseline_symbolic_causal_graphs(self):
        """기본 생물학적/환경적 인과 기호망 초기화"""
        # 1. "생물학적 동맥경화" 기호망
        athero = CausalSymbolNode(
            symbol="동맥경화",
            domain="BIOLOGICAL",
            compressed_causes=["규범/경화 물질 축적", "통로 유연성 상실", "혈관 벽 비대화"],
            compressed_effects=["혈액/에너지 순환 장애", "혈관 내부 압력 폭발 위협", "조직 괴사"]
        )
        athero.relational_ties.extend([
            CausalRelation("혈관 벽 비대화", "동맥경화", "IMPEDES_FLOW", 1.0, "통로 유연성 상실로 흐름 마비"),
            CausalRelation("유입량/혈액 강제 추가", "동맥경화", "AMPLIFIES_PRESSURE", 0.95, "굳어진 혈관에 강제 유입 시 조직 파열 위험"),
            CausalRelation("경화 요소/관성 벽 긁어내기", "동맥경화", "ELIMINATES_OBSTRUCTION", 0.98, "통로 자율성 복원으로 생명력 회복")
        ])
        self.symbol_network["동맥경화"] = athero

        # 2. "가뭄" 기호망
        drought = CausalSymbolNode(
            symbol="가뭄",
            domain="ENVIRONMENTAL",
            compressed_causes=["비/수분 부재", "갈라진 대지", "생명력 건조"],
            compressed_effects=["작물 시듦", "굶주림", "생태계 붕괴 위기"]
        )
        drought.relational_ties.extend([
            CausalRelation("수분 부재", "가뭄", "IMPEDES_FLOW", 1.0, "기초 생명 유지 원천 고갈"),
            CausalRelation("열기 강제 투입", "가뭄", "AMPLIFIES_PRESSURE", 0.9, "건조함 심화로 파열 가속"),
            CausalRelation("근본 단비/원천 수분 공급", "가뭄", "ELIMINATES_OBSTRUCTION", 0.95, "대지 가뭄 해소 및 공진 복원")
        ])
        self.symbol_network["가뭄"] = drought

    def register_symbol(self, node: CausalSymbolNode):
        """신규 인과 상징 기호 등록"""
        self.symbol_network[node.symbol] = node

    def metaphorical_transfer(self, source_symbol: str, target_context: str) -> Dict[str, Any]:
        """
        [은유와 위상적 전이 (Metaphorical Transfer) & 동형성 (Isomorphism)]
        source_symbol의 원형 인과 구조를 target_context(예: "조직 위기")에 그대로 포개어 (Overlay)
        표면적 매체 차이를 상쇄하고 동형의 인과 구조를 추출함.
        """
        if source_symbol not in self.symbol_network:
            raise ValueError(f"Unknown source symbol: {source_symbol}")

        src_node = self.symbol_network[source_symbol]

        # 인과적 불변 구조 (Invariant Structure) 추출
        invariant_structure = {
            "core_mechanism": f"{src_node.symbol} 동형 인과 메커니즘",
            "flow_impediment": src_node.compressed_causes,
            "systemic_crisis": src_node.compressed_effects,
            "isomorphic_invariants": [
                "통로 유연성 상실 (Pathway Rigidity)",
                "흐름의 압박 (Flow Impediment)",
                "시스템 파열 위협 (Systemic Rupture Threat)"
            ]
        }

        # 타겟 맥락에 은유적 투영 (Overlay)
        transferred_diagnosis = {
            "source_symbol": source_symbol,
            "target_context": target_context,
            "isomorphism_detected": True,
            "invariant_structure": invariant_structure,
            "transferred_causes": [f"{target_context} 내 {c}" for c in src_node.compressed_causes],
            "transferred_effects": [f"{target_context} 내 {e}" for e in src_node.compressed_effects]
        }

        self.isomorphic_mappings.append(transferred_diagnosis)
        return transferred_diagnosis

    def evaluate_narrative_coherence(self, diagnosis: Dict[str, Any], candidate_action: str) -> Dict[str, Any]:
        """
        [서사적 결합성 (Narrative Coherence) 기반 의사결정]
        수치 계산 없이 "무엇이 그다음 맥락으로 이어져야 인과적으로 타당한가"라는
        서사적 개연성에 따라 후보 행동의 타당성 및 위험을 판가름함.
        """
        source_symbol = diagnosis["source_symbol"]
        src_node = self.symbol_network[source_symbol]

        # 1. '동맥경화' 동형 체계
        if source_symbol == "동맥경화":
            if "채용" in candidate_action or "자금" in candidate_action or "피 주입" in candidate_action or "인상" in candidate_action:
                return {
                    "action": candidate_action,
                    "is_valid": False,
                    "narrative_verdict": "WRONG_DECISION",
                    "reason": "굳어버린 통로에 유입량(피/자본)을 추가하는 행위는 관성적 압력을 높여 조직 파열을 가속함 (AMPLIFIES_PRESSURE)."
                }
            elif "절차" in candidate_action or "규제" in candidate_action or "폐기" in candidate_action or "긁어내기" in candidate_action:
                return {
                    "action": candidate_action,
                    "is_valid": True,
                    "narrative_verdict": "CORRECT_DECISION",
                    "reason": "관성화된 혈관 벽(보고/승인 절차)을 긁어내어 통로 유연성을 즉시 복원함 (ELIMINATES_OBSTRUCTION)."
                }

        # 2. '가뭄' 동형 체계
        if source_symbol == "가뭄":
            if "압박" in candidate_action or "열기" in candidate_action or "강요" in candidate_action:
                return {
                    "action": candidate_action,
                    "is_valid": False,
                    "narrative_verdict": "WRONG_DECISION",
                    "reason": "건조한 대지에 열기/압박을 더하는 행위는 생명력 소멸을 가속함."
                }
            elif "단비" in candidate_action or "원천" in candidate_action or "휴식" in candidate_action or "수분" in candidate_action:
                return {
                    "action": candidate_action,
                    "is_valid": True,
                    "narrative_verdict": "CORRECT_DECISION",
                    "reason": "원천적 단비(휴식/기초 지원)를 공급하여 대지의 생명력을 회복함."
                }

        return {
            "action": candidate_action,
            "is_valid": False,
            "narrative_verdict": "UNCERTAIN",
            "reason": "서사적 개연성이 불분명함."
        }
