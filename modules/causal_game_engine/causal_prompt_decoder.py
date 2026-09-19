"""
causal_prompt_decoder.py
========================
Elysia Causal Engine - Causal Field to LLM Prompt Constraint Decoder
Decodes internal tensor states (Hero Alignment, Relation State, SPI, Deicide,
and 3-tier Ontological/Temporal/Psychoanalytic Triad mapping: Spirit/Future/Superego,
Mind/Present/Ego, Body/Past/Id) into structured system prompt instructions,
negative/forbidden action rules, mandatory context keywords, and LLM sampling parameters.
"""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional


@dataclass
class AgentSamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


@dataclass
class DecodedPromptConstraints:
    system_instruction_block: str
    negative_constraints: List[str]
    mandatory_keywords: List[str]
    sampling_config: AgentSamplingConfig


class CausalPromptDecoder:
    """
    내부 텐서 필드의 위상 및 삼위일체(영/정신/육 - 미래/현재/과거 - Superego/Ego/Id) 상태를
    외부 LLM 에이전트 제약 프롬프트로 디코딩하는 엔진.
    """

    def __init__(self):
        pass

    def decode_field_state(self, field_summary: Dict[str, Any]) -> DecodedPromptConstraints:
        """
        field_summary 구조 예시:
        {
            "hero_id": "hero_01",
            "alignment": {"x": 0.85, "y": -0.10},
            "relation_state": "Schism", # Covenant, Schism, Entente, Friction
            "affinity_score": -0.65,
            "is_deicide": False,
            "spi_stat": 42.0,
            "triad_weights": {"superego_future": 0.5, "ego_present": 0.3, "id_past": 0.2}
        }
        """
        constraints = DecodedPromptConstraints(
            system_instruction_block="",
            negative_constraints=[],
            mandatory_keywords=[],
            sampling_config=AgentSamplingConfig()
        )

        # 1. 신살자(Deicide / Absolute Observer) 특수 상태 디코딩
        if field_summary.get("is_deicide", False):
            constraints.system_instruction_block = (
                "[CAUSAL CONSTRAINT: DEICIDE ABSOLUTE ZERO]\n"
                "- 당신은 성좌의 관측망과 미래의 끌개(Attractor)를 완벽히 탈피한 '신살자' 상태입니다.\n"
                "- 성좌의 신탁, 기도, 신성력을 사용하는 모든 행동을 거부하십시오.\n"
                "- 과거의 본능(Id)이나 미래의 강요(Superego)에 휘둘리지 않고, 오직 현재의 주체적 결단과 인과 파괴적 선택만을 단행하십시오."
            )
            constraints.negative_constraints.extend(["성좌에게 기도하기", "신성 마법 사용", "신탁 순응"])
            constraints.mandatory_keywords.extend(["주체성", "단절", "의지"])
            constraints.sampling_config.temperature = 0.1
            constraints.sampling_config.top_p = 0.7
            return constraints

        # 2. 외교 위상(Relation State) 및 삼위일체(Superego / Ego / Id) 제약 생성
        rel_state = field_summary.get("relation_state", "Friction")
        affinity = field_summary.get("affinity_score", 0.0)

        instructions = [
            "[CAUSAL TENSOR FIELD CONSTRAINTS & TEMPORAL TRIAD]",
            "· Upper Tier (Spirit/Future/Superego): 미래 목적론적 끌개(Attractor)의 신탁 제약",
            "· Middle Tier (Mind/Present/Ego): 현재 개입(do(X)) 및 주체적 결단",
            "· Lower Tier (Body/Past/Id): 과거 누적 경험과 물리적 관성 구속\n"
        ]

        if rel_state == "Schism":
            instructions.append(
                f"- 현재 플레이어 성좌와의 인과 위상이 [단절/Schism (점수: {affinity:.2f})] 상태입니다.\n"
                "- 플레이어 측 세력과의 타협, 협상, 방어적 동조 행동이 불가능합니다.\n"
                "- 모든 행동 지침은 적대적이고 파괴적인 선제 개입을 지향해야 합니다."
            )
            constraints.negative_constraints.extend(["협상 제시", "동맹 제안", "방어적 후퇴"])
            constraints.mandatory_keywords.extend(["단절", "거부", "파멸"])
            constraints.sampling_config.temperature = 0.85
            constraints.sampling_config.top_p = 0.95

        elif rel_state == "Covenant":
            instructions.append(
                f"- 현재 플레이어 성좌와 [신성 동맹/Covenant (점수: {affinity:.2f})] 결속 상태입니다.\n"
                "- 플레이어의 신탁과 통제 지침(Superego Pull)을 최우선으로 준수하십시오.\n"
                "- 질서적이고 신의를 지키는 엄격한 언행을 유지하십시오."
            )
            constraints.negative_constraints.extend(["배신", "무단 이탈", "독단적 행동"])
            constraints.mandatory_keywords.extend(["맹세", "규율", "수호"])
            constraints.sampling_config.temperature = 0.3
            constraints.sampling_config.top_p = 0.8

        else: # Friction / Entente
            instructions.append(
                f"- 현재 인과 위상이 [{rel_state} (점수: {affinity:.2f})] 상태입니다.\n"
                "- 상위 신탁의 끌개와 과거 경험의 관성 사이에서 실시간 중재와 개입 선택을 탐색하십시오."
            )
            constraints.sampling_config.temperature = 0.6
            constraints.sampling_config.top_p = 0.9

        # 3. 가치관 좌표 표류 기반 세부 지침 주입
        alignment = field_summary.get("alignment", {"x": 0.0, "y": 0.0})
        if alignment["x"] > 0.7 and alignment["y"] < -0.5:  # Lawful Evil 타락 임계점
            instructions.append(
                "- 가치관이 [계약과 악/Lawful Evil] 근방으로 표류했습니다.\n"
                "- 철저한 계산과 정당한 계약의 탈을 쓴 악의적 선택을 유도하십시오."
            )
            constraints.mandatory_keywords.extend(["계약", "대가", "규율"])

        constraints.system_instruction_block = "\n".join(instructions)
        return constraints

    def build_agent_payload(self, field_summary: Dict[str, Any], user_prompt: str) -> Dict[str, Any]:
        """외부 LLM API 호출에 전달할 최종 시스템 payload 생성"""
        decoded = self.decode_field_state(field_summary)

        full_system_prompt = (
            f"{decoded.system_instruction_block}\n\n"
            f"[FORBIDDEN ACTIONS]: {', '.join(decoded.negative_constraints)}\n"
            f"[MANDATORY CONTEXT KEYWORDS]: {', '.join(decoded.mandatory_keywords)}"
        )

        return {
            "system_prompt": full_system_prompt,
            "user_prompt": user_prompt,
            "parameters": {
                "temperature": decoded.sampling_config.temperature,
                "top_p": decoded.sampling_config.top_p,
            }
        }
