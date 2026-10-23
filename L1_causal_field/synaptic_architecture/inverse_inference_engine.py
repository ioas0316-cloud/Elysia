import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from core.memory.causal_controller import CausalMemoryController
from synaptic_architecture.cognitive_field_adapter import CharacterStats, FieldParameters, CognitiveFieldAdapter

@dataclass
class EpistemicFact:
    subject_id: str
    fact_type: str        # "HERO", "VILLAIN", "NEUTRAL"
    confidence: float
    description: str
    context: str

class InverseInferenceEngine:
    """
    [Inverse Inference Engine: 역방향 지각 성찰 관측 엔진]
    순방향 필드 동역학에 의한 물리적 행동(Action: 경비병의 경례, 시민들의 도망침 등)의 붕괴(Collapse)
    결과를 역으로 모니터링하여, 고차원적 가치와 의미론적 팩트(Epistemic Fact)를 역추론 및 도출합니다.
    도출된 팩트는 Causal Engram 형태로 영구히 메모리에 이식되어 다시 거시 사회적 중력장(Phi_social)을 왜곡시킵니다.
    """
    def __init__(self, memory_controller: Optional[CausalMemoryController] = None):
        self.memory_controller = memory_controller if memory_controller is not None else CausalMemoryController()
        self.factual_registry: Dict[str, EpistemicFact] = {}

    def observe_and_infer(self, subject_id: str, collapsed_action: str, context: Dict[str, Any], stats: CharacterStats) -> Optional[EpistemicFact]:
        """
        물리적 행동과 주변 상황 맥락을 역관측하여 캐릭터의 정체성(Hero/Villain)을 역추론합니다.

        [대수적/맥락적 역추론 규칙]
        - Guard_Salute / Open_Gate + Royal_Gate 맥락 => HERO (명예/정통성 인정된 구원자)
        - Citizens_Flee / Guard_Draw_Weapon + Public_Square 맥락 => VILLAIN (사회적 위협 세력)
        """
        fact = None
        action_lower = collapsed_action.lower()
        ctx_name = context.get("location_context", "unknown").lower()
        force_applied = context.get("external_force", 0.0)

        # 1. 완벽한 역방향 추론 (Inverse Inference) 분기 배격 수식화
        # 외력(강요)이 없고 자발적으로 경비병이 머리를 숙이거나 문을 여는 행위 관찰됨
        if ("salute" in action_lower or "open_gate" in action_lower) and force_applied < 1.0:
            fact = EpistemicFact(
                subject_id=subject_id,
                fact_type="HERO",
                confidence=0.9,
                description=f"경비병들이 외력이 없는 자발적 상태에서 {subject_id}에게 허리를 숙이고 성문을 개방하였습니다. 이 극적이고 자발적인 경외의 행동은 {subject_id}가 높은 도덕적 정당성을 획득한 진정한 영웅(Hero)임을 확증합니다.",
                context=f"성문 주변 ({ctx_name})"
            )
            # 순방향 사회적 중력으로 환원: 명예 스탯 자율 상승
            stats.honor += 25.0

        # 외력이 없고 시민들이 공포로 인해 사방으로 흩어져 도망치는 행위 관찰됨
        elif "flee" in action_lower or "draw_weapon" in action_lower:
            fact = EpistemicFact(
                subject_id=subject_id,
                fact_type="VILLAIN",
                confidence=0.95,
                description=f"인접한 시민들이 {subject_id}의 접근을 자각하자마자 자발적 패닉 상태에서 사각 골목으로 사방 분산 도주하였습니다. 이는 {subject_id}가 전장에 발산하는 공포의 악명(Villain)에 대한 100% 인과적 물리 반응입니다.",
                context=f"광장 및 거주구역 ({ctx_name})"
            )
            # 순방향 사회적 중력으로 환원: 악명 스탯 자율 상승
            stats.infamy += 30.0

        if fact:
            self.factual_registry[subject_id] = fact

            # 2. Causal Memory Engram 각인 및 지식 주입
            self.memory_controller.write_causal_engram(
                data_blob={
                    "type": "EPISTEMIC_INVERSE_INFERENCE",
                    "subject_id": subject_id,
                    "fact_type": fact.fact_type,
                    "confidence": fact.confidence,
                    "narrative": fact.description,
                    "location_context": fact.context,
                    "stats_feedback": {
                        "honor_increment": 25.0 if fact.fact_type == "HERO" else 0.0,
                        "infamy_increment": 30.0 if fact.fact_type == "VILLAIN" else 0.0
                    }
                },
                emotional_value=8.0,  # 존재 정체성의 붕괴에 따른 깊은 인지 가중치
                cause_id="InverseInferenceEngine",
                origin_axis="social_epistemic"
            )
            self.memory_controller.flush_index()

        return fact
