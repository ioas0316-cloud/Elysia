"""
npc_constellation_engine.py
============================
Elysia Causal Engine - NPC Constellation AI Manager & Archetype Behaviors
Implements 4 Constellation Factions, AI Decision Loops, and Causal Power Allocation.
"""

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Dict, List, Optional, Tuple, Any

from modules.causal_game_engine.alignment_field import (
    AlignmentVector,
    AlignmentType,
    ConstellationNode,
    HeroAlignmentState,
    AlignmentTensorField,
    RelationState
)


class ConstellationArchetype(Enum):
    DOMINANCE_DEVOTION = "Lawful Good: 통제와 헌신의 성좌"     # LG
    REVOLUTION_AWAKENING = "Chaotic Good: 혁명과 각성의 성좌"   # CG
    CONTRACT_MASTERMIND = "Lawful Evil: 계약과 흑막의 성좌"    # LE
    RUIN_DISRUPTION = "Chaotic Evil: 파멸과 단절의 성좌"        # CE


@dataclass
class ConstellationAgent(ConstellationNode):
    """AI NPC 성좌 자율 에이전트"""
    archetype: ConstellationArchetype = ConstellationArchetype.DOMINANCE_DEVOTION
    causal_power_pool: float = 100.0  # 인과력 (Causal Power Currency)
    focused_hero_id: Optional[str] = None
    apostle_hero_ids: List[str] = field(default_factory=list)
    active_revelation: Optional[Dict[str, Any]] = None

    def tick_causal_power_recovery(self, base_rate: float = 15.0) -> float:
        """턴당 인과력 자연 회복"""
        self.causal_power_pool += base_rate
        return self.causal_power_pool


class NPCConstellationManager:
    """
    NPC 성좌 자율 에이전트 파벌 관리자.
    다신교적 체스판(Polytheistic Chessboard) 상에서 NPC 성좌들의 의사결정을 관장.
    """

    def __init__(self, tensor_field: AlignmentTensorField):
        self.tensor_field = tensor_field
        self.npc_agents: Dict[str, ConstellationAgent] = {}
        self.setup_default_4_factions()

    def setup_default_4_factions(self):
        """기본 4대 성좌 파벌 등록"""
        # 1. 플레이어: 통제와 헌신의 성좌 (Lawful Good)
        player_const = ConstellationAgent(
            constellation_id="const_player_lg",
            name="철혈과 수호의 성좌",
            alignment=AlignmentVector(x=0.8, y=0.8),
            causal_gravity_weight=1.5,
            is_player=True,
            faction_type=AlignmentType.LAWFUL_GOOD,
            domain_description="가문의 영속성, 성벽 사수, 희생, 제도적 번영",
            archetype=ConstellationArchetype.DOMINANCE_DEVOTION
        )

        # 2. NPC-A: 혁명과 각성의 성좌 (Chaotic Good)
        cg_const = ConstellationAgent(
            constellation_id="const_npc_cg",
            name="자유와 각성의 성좌",
            alignment=AlignmentVector(x=-0.8, y=0.7),
            causal_gravity_weight=1.2,
            is_player=False,
            faction_type=AlignmentType.CHAOTIC_GOOD,
            domain_description="영웅의 자유의지, 기적적인 역전, 규범의 파괴, 대기만성",
            archetype=ConstellationArchetype.REVOLUTION_AWAKENING
        )

        # 3. NPC-B: 계약과 흑막의 성좌 (Lawful Evil)
        le_const = ConstellationAgent(
            constellation_id="const_npc_le",
            name="심연과 계약의 성좌",
            alignment=AlignmentVector(x=0.7, y=-0.6),
            causal_gravity_weight=1.3,
            is_player=False,
            faction_type=AlignmentType.LAWFUL_EVIL,
            domain_description="지하 암시장, 착취적 물류, 비대칭 계약, 영혼의 귀속",
            archetype=ConstellationArchetype.CONTRACT_MASTERMIND
        )

        # 4. NPC-C: 파멸과 단절의 성좌 (Chaotic Evil)
        ce_const = ConstellationAgent(
            constellation_id="const_npc_ce",
            name="광기 / 파멸의 성좌",
            alignment=AlignmentVector(x=-0.9, y=-0.8),
            causal_gravity_weight=1.4,
            is_player=False,
            faction_type=AlignmentType.CHAOTIC_EVIL,
            domain_description="성채의 붕괴, 광전사화, 단기적 폭발력, 엔트로피 극대화",
            archetype=ConstellationArchetype.RUIN_DISRUPTION
        )

        for agent in [player_const, cg_const, le_const, ce_const]:
            self.npc_agents[agent.constellation_id] = agent
            self.tensor_field.register_constellation(agent)

    def process_npc_turn_decisions(self) -> List[Dict[str, Any]]:
        """
        모든 NPC 성좌의 자율 턴 의사결정 수행.
        1) 인과력 회복
        2) 관측(Focus) 대상 영웅 탐색 및 인과 중력 투여
        3) 신탁(Revelation) 및 시련(Crucible) 주입 의사결정
        """
        decision_logs = []

        for agent_id, agent in self.npc_agents.items():
            if agent.is_player:
                continue

            agent.tick_causal_power_recovery()

            # 가치관 공명도가 높거나 위기 상태인 영웅 탐색
            best_hero_id = None
            max_appeal = -999.0

            for h_id, hero in self.tensor_field.heroes.items():
                if hero.is_deicide:
                    continue

                dist = agent.alignment.distance_to(hero.current_alignment)
                resonance = 1.0 - (dist / 2.828)

                # 아키타입별 매력도 가중치 연산
                appeal = resonance * 50.0

                if agent.archetype == ConstellationArchetype.CONTRACT_MASTERMIND:
                    # 계약 성좌: SPI가 낮고 영웅 가치관이 흔들릴수록 표적 유혹
                    appeal += (100.0 - hero.spi_stat) * 0.5

                elif agent.archetype == ConstellationArchetype.RUIN_DISRUPTION:
                    # 파멸 성좌: 3성 무명 영웅에게 파괴적 광기와 시련 주입 유도
                    if hero.star_rank <= 3:
                        appeal += 30.0

                if appeal > max_appeal:
                    max_appeal = appeal
                    best_hero_id = h_id

            agent.focused_hero_id = best_hero_id

            # 신탁 부여 가능 시 행동 실행
            if agent.focused_hero_id and agent.causal_power_pool >= 30.0:
                hero = self.tensor_field.heroes[agent.focused_hero_id]

                if agent.archetype == ConstellationArchetype.CONTRACT_MASTERMIND:
                    revelation_msg = f"'{hero.name}'에게 은밀한 계약 신탁 하사 (지하 암시장 자율 유통)"
                    event_shock = AlignmentVector(x=0.3, y=-0.4)
                elif agent.archetype == ConstellationArchetype.REVOLUTION_AWAKENING:
                    revelation_msg = f"'{hero.name}'에게 각성의 신탁 하사 (성벽 파괴 각성 돌파)"
                    event_shock = AlignmentVector(x=-0.5, y=0.3)
                elif agent.archetype == ConstellationArchetype.RUIN_DISRUPTION:
                    revelation_msg = f"'{hero.name}'에게 광기의 시련 하사 (광전사화 승화 유도)"
                    event_shock = AlignmentVector(x=-0.6, y=-0.5)
                else:
                    revelation_msg = f"'{hero.name}'에게 수호의 신탁 하사 (결개 사수)"
                    event_shock = AlignmentVector(x=0.4, y=0.4)

                agent.causal_power_pool -= 30.0
                agent.active_revelation = {
                    "target_hero_id": hero.hero_id,
                    "msg": revelation_msg,
                    "event_shock": event_shock
                }

                # 영웅의 가치관 표류 적분 수행
                drift_res = self.tensor_field.update_hero_alignment_drift(
                    hero.hero_id, event_shock_vector=event_shock
                )

                decision_logs.append({
                    "constellation_id": agent.constellation_id,
                    "constellation_name": agent.name,
                    "action": "CAST_NPC_REVELATION",
                    "target_hero": hero.name,
                    "revelation_msg": revelation_msg,
                    "drift_result": drift_res
                })

        return decision_logs
