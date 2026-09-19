"""
causal_interference.py
======================
Elysia Causal Engine - Causal Interference & Heretical Artifact Synthesis
Handles dual/multi-observation focus interference, phase distortion,
and synthesis of Heretical Artifacts (이단적 보구).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math

from modules.causal_game_engine.alignment_field import (
    AlignmentVector,
    HeroAlignmentState,
    AlignmentTensorField,
    ConstellationNode
)


@dataclass
class HereticalArtifact:
    """두 성좌의 위상 간섭 속성이 기괴하게 융합된 이단적 보구"""
    id: str
    name: str
    primary_constellation_id: str
    secondary_constellation_id: str
    interference_intensity: float  # 간섭 강도 [0.0 ~ 1.0]
    hybrid_domain_power: str
    atypical_side_effect: str
    bonus_attack: float = 50.0
    bonus_defense: float = 30.0


class CausalInterferenceEngine:
    """
    인과장 중첩 및 간섭 엔진.
    플레이어와 NPC 성좌가 동일 노드/영웅을 동시에 관측(Focus)할 때
    인과 중력이 증폭됨과 동시에 위상 상충에 의한 왜곡 파동이 발생하여
    '이단적 보구(Heretical Artifact)'를 연성함.
    """

    def __init__(self, tensor_field: AlignmentTensorField):
        self.tensor_field = tensor_field

    def evaluate_multi_observation_focus(
        self,
        hero: HeroAlignmentState,
        focusing_constellation_ids: List[str]
    ) -> Dict[str, Any]:
        """
        다중 관측에 의한 인과장 중첩 및 간섭도 연산.
        """
        if len(focusing_constellation_ids) < 2:
            return {
                "interference_active": False,
                "amplified_gravity": 1.0,
                "phase_distortion": 0.0,
                "heretical_artifact": None
            }

        const_1 = self.tensor_field.constellations[focusing_constellation_ids[0]]
        const_2 = self.tensor_field.constellations[focusing_constellation_ids[1]]

        dist = const_1.alignment.distance_to(const_2.alignment)

        # 인과 중력 증폭: 다중 관측으로 인해 인과적 임계점 누적 속도가 2배~3배로 증폭
        amplified_gravity = float(1.0 + (const_1.causal_gravity_weight + const_2.causal_gravity_weight) * 0.8)

        # 위상 왜곡도 (Phase Distortion): 성좌 간 가치관 거리가 멀수록 극대화
        phase_distortion = float(dist / 2.828)

        # 위상 왜곡도가 0.3 이상이면 '이단적 보구' 연성 조건 성립
        heretical_artifact = None
        if phase_distortion >= 0.25:
            heretical_artifact = self._synthesize_heretical_artifact(
                hero, const_1, const_2, phase_distortion
            )

        return {
            "interference_active": True,
            "focusing_constellations": [const_1.name, const_2.name],
            "amplified_gravity": round(amplified_gravity, 2),
            "phase_distortion": round(phase_distortion, 4),
            "heretical_artifact": heretical_artifact
        }

    def _synthesize_heretical_artifact(
        self,
        hero: HeroAlignmentState,
        c1: ConstellationNode,
        c2: ConstellationNode,
        distortion: float
    ) -> HereticalArtifact:
        """이단적 보구 연성 로직"""
        artifact_id = f"heretical_{c1.constellation_id[:4]}_{c2.constellation_id[:4]}_{hero.hero_id}"
        artifact_name = f"[{c1.name[:2]} × {c2.name[:2]}] 이단적 보구: 위상 왜곡의 붉은 신인(神印)"

        hybrid_power = f"<{c1.domain_description[:10]}>의 수호권능과 <{c2.domain_description[:10]}>의 파괴력이 기괴하게 상충 중첩됨"
        side_effect = f"공격 시 20% 확률로 성채 전체 민심 소폭 감소 및 순간 300% 폭발적 공격력 발산 (위상 왜곡도: {distortion:.2f})"

        return HereticalArtifact(
            id=artifact_id,
            name=artifact_name,
            primary_constellation_id=c1.constellation_id,
            secondary_constellation_id=c2.constellation_id,
            interference_intensity=distortion,
            hybrid_domain_power=hybrid_power,
            atypical_side_effect=side_effect,
            bonus_attack=85.0,
            bonus_defense=45.0
        )
