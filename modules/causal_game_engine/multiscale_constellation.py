"""
multiscale_constellation.py
===========================
Elysia Causal Engine - Multi-Scale Cosmological Hierarchy & Causal Mechanics
Implements Tier 1 (Primordial Principles), Tier 2 (Major Pantheons), and
Tier 3 (Subordinated, Wandering, Heretical Deities) along with Causal Subcontracting,
Causal Tax, Causal Share, Usurpation, and Ascension Mechanics.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple, Any

from modules.causal_game_engine.alignment_field import (
    AlignmentVector,
    AlignmentType,
    ConstellationNode,
    HeroAlignmentState,
    AlignmentTensorField
)


class ConstellationTier(Enum):
    TIER_1_PRIMORDIAL = "Tier 1: 근원적 주권 / 법칙"
    TIER_2_PANTHEON = "Tier 2: 세력 신좌 / 파벌"
    TIER_3_LESSER = "Tier 3: 개별 / 방랑 / 이단 성좌"


class Tier3AffiliationType(Enum):
    SUBORDINATED_VASSAL = "종속 신격 (Subordinated Vassal)"
    WANDERING_UNAFFILIATED = "방랑 / 무소속 (Wandering Deities)"
    HERETICAL_OUTLAW = "이단 / 심연 (Heretical Outlaw)"


@dataclass
class Tier1PrimordialPrinciple:
    """
    Tier 1: 근원적 주권 / 법칙 (Primordial Principles)
    우주의 거시적 추상 개념이자 절대 물리 법칙 그 자체.
    """
    principle_id: str
    name: str  # 예: '엔트로피의 섭리', '인과적 정합의 원형'
    description: str
    global_causal_pool: float = 10000.0  # C_global
    entropy_coefficient: float = 0.05
    boundary_constraints: Dict[str, Any] = field(default_factory=dict)

    def emit_base_field_energy(self, amount: float = 500.0) -> float:
        """거시 인과장 에너지 배분"""
        if self.global_causal_pool >= amount:
            self.global_causal_pool -= amount
            return amount
        drained = self.global_causal_pool
        self.global_causal_pool = 0.0
        return drained


@dataclass
class MultiscaleConstellationNode(ConstellationNode):
    """
    상하위 스케일을 지니는 다층 성좌 노드.
    """
    tier: ConstellationTier = ConstellationTier.TIER_3_LESSER
    affiliation_type: Tier3AffiliationType = Tier3AffiliationType.WANDERING_UNAFFILIATED

    # Tier 2 상위 신좌 정보 (종속 성좌일 경우 지정)
    parent_pantheon_id: Optional[str] = None
    subordinated_vassal_ids: List[str] = field(default_factory=list)

    # 인과율 자원 및 지분
    causal_power_pool: float = 100.0
    causal_share: float = 0.05  # [0.0 ~ 1.0] 전체 우주 인과율 지분율
    causal_tax_rate: float = 0.15  # 종속 성좌가 상위 신좌에 바치는 인과율 수수료 비율

    # 방랑 성좌 특수 미학 (Aesthetics)
    aesthetics_criteria: Optional[Dict[str, Any]] = None

    # 이단 성좌 특수 위법적 승화 카운트
    heretical_ascension_count: int = 0


class MultiscaleCosmologicalEngine:
    """
    3단계 위계 구조 및 상호작용 메카닉 관리자.
    """

    def __init__(self, tensor_field: AlignmentTensorField):
        self.tensor_field = tensor_field
        self.tier1_principles: Dict[str, Tier1PrimordialPrinciple] = {}
        self.constellations: Dict[str, MultiscaleConstellationNode] = {}
        self.setup_default_cosmology()

    def setup_default_cosmology(self):
        """기본 3단계 위계 우주론 초기화"""
        # 1. Tier 1 근원적 법칙 등록
        t1_entropy = Tier1PrimordialPrinciple(
            principle_id="t1_entropy",
            name="엔트로피의 섭리",
            description="우주 만물의 쇠퇴와 에너지 전이의 절대 법칙",
            global_causal_pool=10000.0
        )
        t1_causality = Tier1PrimordialPrinciple(
            principle_id="t1_causality",
            name="인과적 정합의 원형",
            description="원인 없는 결과가 존재할 수 없게 구동하는 법칙",
            global_causal_pool=10000.0
        )
        self.tier1_principles[t1_entropy.principle_id] = t1_entropy
        self.tier1_principles[t1_causality.principle_id] = t1_causality

        # 2. Tier 2 세력 신좌 등록
        t2_lg = MultiscaleConstellationNode(
            constellation_id="t2_pantheon_lg",
            name="통제와 헌신의 신좌",
            alignment=AlignmentVector(x=0.8, y=0.8),
            causal_gravity_weight=2.5,
            tier=ConstellationTier.TIER_2_PANTHEON,
            faction_type=AlignmentType.LAWFUL_GOOD,
            causal_power_pool=2000.0,
            causal_share=0.40,
            is_player=False
        )

        t2_le = MultiscaleConstellationNode(
            constellation_id="t2_pantheon_le",
            name="계약과 흑막의 주권",
            alignment=AlignmentVector(x=0.7, y=-0.7),
            causal_gravity_weight=2.2,
            tier=ConstellationTier.TIER_2_PANTHEON,
            faction_type=AlignmentType.LAWFUL_EVIL,
            causal_power_pool=1800.0,
            causal_share=0.35,
            is_player=False
        )

        # 3. Tier 3 하위 / 방랑 / 이단 성좌 등록
        t3_iron = MultiscaleConstellationNode(
            constellation_id="t3_vassal_iron",
            name="철혈 수성의 성좌",
            alignment=AlignmentVector(x=0.75, y=0.75),
            causal_gravity_weight=1.0,
            tier=ConstellationTier.TIER_3_LESSER,
            affiliation_type=Tier3AffiliationType.SUBORDINATED_VASSAL,
            parent_pantheon_id="t2_pantheon_lg",
            causal_power_pool=150.0,
            causal_share=0.08,
            causal_tax_rate=0.20
        )
        t2_lg.subordinated_vassal_ids.append("t3_vassal_iron")

        t3_tragedy = MultiscaleConstellationNode(
            constellation_id="t3_wandering_tragedy",
            name="비극적 몰락을 관람하는 성좌",
            alignment=AlignmentVector(x=-0.2, y=-0.3),
            causal_gravity_weight=0.8,
            tier=ConstellationTier.TIER_3_LESSER,
            affiliation_type=Tier3AffiliationType.WANDERING_UNAFFILIATED,
            causal_power_pool=100.0,
            causal_share=0.05,
            aesthetics_criteria={"target_star_rank": 3, "trigger_condition": "TRAGIC_FALL"}
        )

        t3_heretic = MultiscaleConstellationNode(
            constellation_id="t3_heretical_abyss",
            name="심연의 해커 성좌",
            alignment=AlignmentVector(x=-0.9, y=-0.9),
            causal_gravity_weight=1.2,
            tier=ConstellationTier.TIER_3_LESSER,
            affiliation_type=Tier3AffiliationType.HERETICAL_OUTLAW,
            causal_power_pool=200.0,
            causal_share=0.07
        )

        # 플레이어 초기 상태: Tier 3 방랑 성좌
        player_const = MultiscaleConstellationNode(
            constellation_id="const_player",
            name="새벽의 관측자 (플레이어)",
            alignment=AlignmentVector(x=0.5, y=0.5),
            causal_gravity_weight=1.5,
            tier=ConstellationTier.TIER_3_LESSER,
            affiliation_type=Tier3AffiliationType.WANDERING_UNAFFILIATED,
            causal_power_pool=300.0,
            causal_share=0.05,
            is_player=True
        )

        for const in [t2_lg, t2_le, t3_iron, t3_tragedy, t3_heretic, player_const]:
            self.register_multiscale_constellation(const)

    def register_multiscale_constellation(self, node: MultiscaleConstellationNode):
        self.constellations[node.constellation_id] = node
        self.tensor_field.register_constellation(node)

    def execute_causal_subcontract(
        self, parent_id: str, vassal_id: str, requested_power: float
    ) -> Dict[str, Any]:
        """
        [상하위 스케일 역학 ①] 인과율 하청 및 위임 (Causal Subcontracting)
        Tier 2 상위 성좌가 Tier 3 종속 성좌에게 인과율 자원을 전송.
        """
        parent = self.constellations.get(parent_id)
        vassal = self.constellations.get(vassal_id)

        if not parent or not vassal:
            return {"success": False, "reason": "Constellation not found"}

        if parent.tier != ConstellationTier.TIER_2_PANTHEON:
            return {"success": False, "reason": "Parent is not a Tier 2 Pantheon"}

        if vassal.parent_pantheon_id != parent_id:
            return {"success": False, "reason": "Vassal is not subordinated to parent"}

        transfer_amount = min(parent.causal_power_pool, requested_power)
        parent.causal_power_pool -= transfer_amount
        vassal.causal_power_pool += transfer_amount

        return {
            "success": True,
            "parent_id": parent_id,
            "vassal_id": vassal_id,
            "transferred_power": transfer_amount,
            "parent_remaining_pool": parent.causal_power_pool,
            "vassal_new_pool": vassal.causal_power_pool
        }

    def process_revelation_with_tax(
        self, constellation_id: str, hero_id: str, power_cost: float
    ) -> Dict[str, Any]:
        """
        Tier 3 종속 성좌가 지상계 영웅에게 신탁을 내릴 때
        상위 성좌에게 인과율 수수료(Causal Tax)를 지불함.
        """
        node = self.constellations.get(constellation_id)
        if not node:
            return {"success": False, "reason": "Constellation not found"}

        tax = 0.0
        if node.affiliation_type == Tier3AffiliationType.SUBORDINATED_VASSAL and node.parent_pantheon_id:
            parent = self.constellations.get(node.parent_pantheon_id)
            if parent:
                tax = power_cost * node.causal_tax_rate
                parent.causal_power_pool += tax

        total_cost = power_cost + tax
        if node.causal_power_pool < total_cost:
            return {"success": False, "reason": "Insufficient causal power including tax"}

        node.causal_power_pool -= total_cost

        return {
            "success": True,
            "constellation_id": constellation_id,
            "hero_id": hero_id,
            "power_cost": power_cost,
            "causal_tax_paid": tax,
            "remaining_power": node.causal_power_pool
        }

    def process_illegal_ascension(self, heretic_id: str, hero_id: str) -> Dict[str, Any]:
        """
        [Tier 3 이단 성좌 메카닉] '위법적 승화' (Illegal Ascension)
        규칙을 뛰어넘는 인과적 폭발력을 주는 대신, 영웅 존재 자체를 인과계에서 삭제.
        """
        heretic = self.constellations.get(heretic_id)
        hero = self.tensor_field.heroes.get(hero_id)

        if not heretic or not hero:
            return {"success": False, "reason": "Heretic constellation or hero not found"}

        if heretic.affiliation_type != Tier3AffiliationType.HERETICAL_OUTLAW:
            return {"success": False, "reason": "Node is not a Heretical/Outlaw constellation"}

        # 영웅 존재 삭제 (Deicide 이상의 탈태)
        hero.is_deicide = True
        hero.current_alignment = AlignmentVector(0.0, 0.0)
        heretic.heretical_ascension_count += 1
        heretic.causal_share += 0.05

        return {
            "success": True,
            "heretic_id": heretic_id,
            "erased_hero_id": hero_id,
            "hero_name": hero.name,
            "heretic_new_causal_share": heretic.causal_share,
            "description": f"이단 성좌 '{heretic.name}'이(가) 영웅 '{hero.name}'의 존재를 인과계에서 금지된 법칙으로 강제 소멸시켰습니다."
        }

    def check_and_trigger_ascension(
        self, constellation_id: str, share_boost: float = 0.0
    ) -> Dict[str, Any]:
        """
        [상하위 스케일 역학 ②] 하극상 및 승격 (Usurpation & Ascension)
        Tier 3 성좌의 인과적 지분(Causal Share)이 임계점(>= 0.30)에 도달하면 Tier 2 주권 신좌로 승격!
        """
        node = self.constellations.get(constellation_id)
        if not node:
            return {"success": False, "reason": "Constellation not found"}

        node.causal_share += share_boost

        if node.tier == ConstellationTier.TIER_3_LESSER and node.causal_share >= 0.30:
            # Tier 2 주권 신좌로 승격!
            node.tier = ConstellationTier.TIER_2_PANTHEON
            node.causal_gravity_weight *= 2.0

            # 만약 종속 성좌였다면 상위 신좌의 구속에서 독립
            if node.parent_pantheon_id:
                parent = self.constellations.get(node.parent_pantheon_id)
                if parent and node.constellation_id in parent.subordinated_vassal_ids:
                    parent.subordinated_vassal_ids.remove(node.constellation_id)
                node.parent_pantheon_id = None

            node.affiliation_type = Tier3AffiliationType.WANDERING_UNAFFILIATED

            return {
                "ascended": True,
                "constellation_id": node.constellation_id,
                "name": node.name,
                "new_tier": node.tier.value,
                "new_causal_share": node.causal_share,
                "description": f"성좌 '{node.name}'이(가) 우주적 인과 지분을 장악하여 새로운 Tier 2 주권 신좌로 승격했습니다!"
            }

        return {
            "ascended": False,
            "constellation_id": node.constellation_id,
            "current_causal_share": node.causal_share,
            "threshold": 0.30
        }
