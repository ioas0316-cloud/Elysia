"""
alignment_field.py
==================
Elysia Causal Engine - 2D Alignment Tensor Field & Drift Mechanics
Integrates D&D Alignment Tensor Space with Constellation Phase Lock Dynamics.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple, Any


class AlignmentType(Enum):
    LAWFUL_GOOD = "Lawful Good (통제와 헌신)"
    CHAOTIC_GOOD = "Chaotic Good (혁명과 각성)"
    LAWFUL_EVIL = "Lawful Evil (계약과 흑막)"
    CHAOTIC_EVIL = "Chaotic Evil (파멸과 단절)"
    TRUE_NEUTRAL = "True Neutral (절대 중립 / 신살자)"


class RelationState(Enum):
    COVENANT = "Covenant"  # S >= 0.65 (신성 동맹)
    ENTENTE = "Entente"    # 0.20 <= S < 0.65 (실리적 조약)
    FRICTION = "Friction"  # -0.40 <= S < 0.20 (위상 마찰)
    SCHISM = "Schism"      # S < -0.40 (신좌 대리전)


@dataclass
class AlignmentVector:
    """2차원 가치관 좌표 벡터 (x: Law-Chaos, y: Good-Evil)"""
    x: float  # [-1.0 (Chaos) ~ +1.0 (Law)]
    y: float  # [-1.0 (Evil)  ~ +1.0 (Good)]

    def clamp(self) -> None:
        """좌표 범위를 [-1.0, 1.0]으로 제약"""
        self.x = max(-1.0, min(1.0, float(self.x)))
        self.y = max(-1.0, min(1.0, float(self.y)))

    def distance_to(self, other: "AlignmentVector") -> float:
        """유클리드 거리 연산"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def to_tuple(self) -> Tuple[float, float]:
        return (round(self.x, 4), round(self.y, 4))


@dataclass
class ConstellationNode:
    """성좌(관측자) 엔티티"""
    constellation_id: str
    name: str
    alignment: AlignmentVector
    causal_gravity_weight: float = 1.0  # W_k
    is_player: bool = False
    faction_type: AlignmentType = AlignmentType.TRUE_NEUTRAL
    domain_description: str = ""


@dataclass
class HeroAlignmentState:
    """영웅의 가치관 표류 및 신념 상태"""
    hero_id: str
    name: str
    current_alignment: AlignmentVector  # H(t)
    base_anchor_alignment: AlignmentVector  # H_0 (기본 신념)
    spi_stat: float = 50.0  # 정신력 (SPI)
    star_rank: int = 3
    trajectory_history: List[Tuple[float, float]] = field(default_factory=list)

    # 미분 방정식 계수
    alpha_gravity_sensitivity: float = 0.05
    beta_event_sensitivity: float = 0.10
    gamma_spi_resilience: float = 0.02

    is_deicide: bool = False  # 신살자(Deicide) 각성 여부
    bound_constellation_id: Optional[str] = None  # 주 후원 성좌 ID

    def record_history(self) -> None:
        """현재 가치관 좌표 기록 (최대 12틱 유지)"""
        self.trajectory_history.append((self.current_alignment.x, self.current_alignment.y))
        if len(self.trajectory_history) > 12:
            self.trajectory_history.pop(0)


class AlignmentTensorField:
    """
    2차원 가치관 텐서 필드 및 가치관 표류 적분 연산 엔진.
    """

    def __init__(self):
        self.constellations: Dict[str, ConstellationNode] = {}
        self.heroes: Dict[str, HeroAlignmentState] = {}

    def register_constellation(self, constellation: ConstellationNode) -> None:
        self.constellations[constellation.constellation_id] = constellation

    def register_hero(self, hero: HeroAlignmentState) -> None:
        self.heroes[hero.hero_id] = hero
        hero.record_history()

    def calculate_constellation_affinity(self, id1: str, id2: str) -> Tuple[float, RelationState]:
        """
        두 성좌 간의 친밀도 S_ij 및 외교 상태 연산
        S_ij = 1.0 - (Distance / sqrt(2))
        """
        c1 = self.constellations[id1]
        c2 = self.constellations[id2]
        dist = c1.alignment.distance_to(c2.alignment)
        affinity = 1.0 - (dist / math.sqrt(2.0))

        if affinity >= 0.65:
            state = RelationState.COVENANT
        elif affinity >= 0.20:
            state = RelationState.ENTENTE
        elif affinity >= -0.40:
            state = RelationState.FRICTION
        else:
            state = RelationState.SCHISM

        return float(affinity), state

    def update_hero_alignment_drift(
        self,
        hero_id: str,
        event_shock_vector: Optional[AlignmentVector] = None
    ) -> Dict[str, Any]:
        """
        가치관 표류 미분 방정식 틱 적분 수행:
        dH/dt = alpha * sum(W_k * (A_k - H)) + beta * E_event - gamma * SPI * (H - H_0)
        """
        hero = self.heroes[hero_id]

        # 신살자 상태일 경우 관측장 미분 적분 무효화
        if hero.is_deicide:
            return {
                "hero_id": hero.hero_id,
                "status": "DEICIDE_IMMUNE",
                "dh_vector": (0.0, 0.0),
                "new_alignment": (0.0, 0.0),
                "triggered_event": None
            }

        H = hero.current_alignment
        H0 = hero.base_anchor_alignment

        # 1. 성좌 인과장 끌림 항 연산
        gravity_dx = 0.0
        gravity_dy = 0.0
        for const in self.constellations.values():
            W_k = const.causal_gravity_weight
            A_k = const.alignment
            gravity_dx += W_k * (A_k.x - H.x)
            gravity_dy += W_k * (A_k.y - H.y)

        dx_gravity = hero.alpha_gravity_sensitivity * gravity_dx
        dy_gravity = hero.alpha_gravity_sensitivity * gravity_dy

        # 2. 시련/환경 충격 항 연산
        dx_event = 0.0
        dy_event = 0.0
        if event_shock_vector:
            dx_event = hero.beta_event_sensitivity * event_shock_vector.x
            dy_event = hero.beta_event_sensitivity * event_shock_vector.y

        # 3. SPI 인격적 복원력 항 연산
        dx_resilience = hero.gamma_spi_resilience * (hero.spi_stat / 50.0) * (H0.x - H.x)
        dy_resilience = hero.gamma_spi_resilience * (hero.spi_stat / 50.0) * (H0.y - H.y)

        # 총 미분량 dH/dt 산출 및 위치 업데이트
        dh_x = dx_gravity + dx_event + dx_resilience
        dh_y = dy_gravity + dy_event + dy_resilience

        H.x += dh_x
        H.y += dh_y
        H.clamp()

        hero.record_history()

        # 서사 이벤트 트리거 평가
        event_trigger = self._evaluate_alignment_events(hero)

        return {
            "hero_id": hero.hero_id,
            "dh_vector": (round(dh_x, 4), round(dh_y, 4)),
            "new_alignment": H.to_tuple(),
            "triggered_event": event_trigger
        }

    def _evaluate_alignment_events(self, hero: HeroAlignmentState) -> Optional[Dict[str, Any]]:
        """영웅의 좌표 변화에 따른 서사 이벤트 트리거 평가"""
        H = hero.current_alignment

        # 1. 플레이어 성좌와의 공명 각성 검증
        player_const = next((c for c in self.constellations.values() if c.is_player), None)
        if player_const and H.distance_to(player_const.alignment) <= 0.18:
            return {
                "event_type": "HOLY_RESONANCE_AWAKENING",
                "constellation_id": player_const.constellation_id,
                "description": f"{hero.name}이(가) 플레이어 성좌 '{player_const.name}'과 완전한 동위상 공명을 이루어 현신(Avatar)으로 각성했습니다!"
            }

        # 2. 적대/NPC 성좌로의 배교 검증
        for const in self.constellations.values():
            if not const.is_player and H.distance_to(const.alignment) <= 0.15:
                if hero.bound_constellation_id != const.constellation_id:
                    return {
                        "event_type": "APOSTASY_EVENT",
                        "target_constellation_id": const.constellation_id,
                        "description": f"{hero.name}이(가) 성좌 '{const.name}'의 가치관으로 배교(Apostasy)하여 사도로 전향했습니다!"
                    }

        # 3. 신살자(Deicide) 각성 검증 (정신력 붕괴 및 충돌 지점에서 인과 닻 파괴)
        if abs(H.x) < 0.08 and abs(H.y) < 0.08 and hero.spi_stat >= 80.0:
            hero.is_deicide = True
            H.x, H.y = 0.0, 0.0
            return {
                "event_type": "DEICIDE_AWAKENING",
                "description": f"{hero.name}이(가) 모든 성좌의 관측과 신탁에 환멸을 느끼고 인과적 닻을 끊어내며 '신살자(Deicide)'로 각성했습니다!"
            }

        return None
