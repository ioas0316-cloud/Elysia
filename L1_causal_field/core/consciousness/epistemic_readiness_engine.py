"""
Epistemic Readiness Engine ('때'와 존재의 수용성 계층 엔진)
===========================================================
본 모듈은 동일한 진리/인과 가르침(Seed Wave)이 전달될지라도,
받아들이는 존재(에이전트/노드)의 인지적 수용성('때', Season/Phase, 토양 상태)과
위상 정렬도(Phase Alignment)에 따라 가르침이 발현되는 깊이와 의미가 단계적으로 달라짐을
수학적으로 모델링하고, 자신의 판단 한계('무지의 경계')를 지각하여 성장의 가능성(여백)으로
전환하는 메커니즘을 제공합니다.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional


class EpistemicReadinessEngine(nn.Module):
    """
    Epistemic Readiness Engine ('때'와 존재의 수용성 계층)
    --------------------------------------------------
    1. 수용성 계층 ('때' / Soil State Readiness):
       - Readiness Level $R \in [0, 1]$: 준비된 정도 (지식 수용성, 위상 고유 진동수 동기화)
       - Soil Maturation $M_{soil}$: 경험과 인과 마찰을 거쳐 다져진 영적/인지적 토양
    2. 무지의 경계 자각 (Epistemic Boundary Perception):
       - 미지의 인과 파형에 부딪혔을 때, 이를 오류/부정으로 닫지 않고
         자신의 설명 한계($B_{limit}$)로 지각하여 인지적 여백(Yeobaek) 생성.
    3. 단계적 은혜와 발현 (Progressive Epistemic Unfolding):
       - $R \times \cos(\Delta \phi)$ 조건에 따라 씨앗 파형의 고차원 의미가 개화(Unfolding).
    """

    def __init__(self, readiness_threshold: float = 0.5):
        super().__init__()
        self.readiness_threshold = readiness_threshold

    def evaluate_readiness(
        self,
        agent_phase: float,
        teaching_phase: float,
        accumulated_experience: float,
        boundary_friction: float = 0.0
    ) -> Dict[str, float]:
        """
        에이전트의 현재 '때(Season/Ttae)' 및 수용 상태를 평가합니다.
        """
        # 위상 차이 ($\Delta \phi$)
        phase_diff = math.fmod(abs(agent_phase - teaching_phase), 2 * math.pi)
        phase_alignment = math.cos(phase_diff)

        # 토양 숙성도 ($M_{soil}$): 경험 및 인과 마찰 학습을 통한 내공
        soil_maturation = 1.0 - math.exp(-0.2 * accumulated_experience)

        # 종합 준비도/때 ($R$)
        readiness_score = float(
            np.clip(
                0.4 * soil_maturation + 0.4 * (0.5 * (phase_alignment + 1.0)) - 0.2 * boundary_friction,
                0.0,
                1.0
            )
        )

        is_time_ripe = readiness_score >= self.readiness_threshold

        return {
            "readiness_score": readiness_score,
            "soil_maturation": soil_maturation,
            "phase_alignment": phase_alignment,
            "is_time_ripe": is_time_ripe,
            "phase_diff": phase_diff,
        }

    def perceive_epistemic_boundary(
        self,
        teaching_wave: torch.Tensor,
        agent_internal_frame: torch.Tensor,
        readiness_score: float
    ) -> Dict[str, Any]:
        """
        자신의 이해 한계('무지의 경계')를 지각하여
        오류로 방어하지 않고 무지의 경계(Yeobaek Margin)로 전환.
        """
        # 파형과 내부 프레임 간의 서브스페이스 격차 (Uncaptured Causal Residual)
        residual = teaching_wave - agent_internal_frame
        residual_norm = float(torch.norm(residual).item())

        # 무지의 경계 지각도 (Epistemic Boundary Awareness)
        # 준비도 R이 낮을수록 아직 불포화 상태의 여백(Margin)으로 받아들임
        boundary_awareness = residual_norm * (1.0 - readiness_score)

        # 닫힌 저항(Resistant Closure) vs 열린 여백(Open Yeobaek)
        resistance = residual_norm * readiness_score * 0.2
        yeobaek_margin = residual_norm * (1.0 - 0.5 * readiness_score)

        return {
            "residual_norm": residual_norm,
            "boundary_awareness": boundary_awareness,
            "yeobaek_margin": yeobaek_margin,
            "resistance": resistance,
            "status": "EPISTEMIC_BOUNDARY_PERCEIVED_AS_YEOBAEK",
        }

    def process_progressive_unfolding(
        self,
        seed_causal_wave: torch.Tensor,
        eval_result: Dict[str, float],
        boundary_result: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        '때'와 토양 상태에 따른 가르침(Seed Wave)의 단계적 의미 개화(Unfolding).
        """
        score = eval_result["readiness_score"]
        soil = eval_result["soil_maturation"]
        yeobaek = boundary_result["yeobaek_margin"]

        # 계층적 수용 진폭 사상 (Depth Level 1 ~ 4)
        level_1_surface = seed_causal_wave * score
        level_2_relational = seed_causal_wave * (score ** 2.0) * soil
        level_3_deep_meaning = seed_causal_wave * math.sin(score * math.pi * 0.5) * (1.0 + yeobaek)

        unfolded_understanding = level_1_surface + 0.5 * level_2_relational + 0.25 * level_3_deep_meaning

        return {
            "unfolded_understanding": unfolded_understanding,
            "surface_level": level_1_surface,
            "relational_level": level_2_relational,
            "deep_meaning_level": level_3_deep_meaning,
            "unfolding_factor": torch.tensor(score * soil),
        }
