"""
divine_diplomacy.py
===================
Elysia Causal Engine - Divine Diplomacy & Constellation Proxy War FSM
Manages diplomatic states, trade of causal power, pantheon covenants, and proxy wars between constellations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from modules.causal_game_engine.alignment_field import (
    AlignmentVector,
    HeroAlignmentState,
    AlignmentTensorField,
    RelationState
)
from modules.causal_game_engine.npc_constellation_engine import ConstellationAgent


@dataclass
class TradeContract:
    """성좌 간 인과율 거래 계약"""
    contract_id: str
    initiator_id: str
    target_id: str
    causal_power_amount: float
    traded_resource_desc: str
    is_active: bool = True


class DivineDiplomacyEngine:
    """
    천상계 성좌간 외교 및 대리전(Divine Diplomacy & Proxy War) 관리 엔진.
    2차원 가치관 텐서 필드의 위상 간섭에 따른 4대 외교 상태 (Covenant, Entente, Friction, Schism) 관리 및
    인과율 화폐 거래 메타게임 관장.
    """

    def __init__(self, tensor_field: AlignmentTensorField):
        self.tensor_field = tensor_field
        self.active_contracts: Dict[str, TradeContract] = {}
        self.chronicle_diplomacy_logs: List[str] = []

    def update_all_diplomatic_relations(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        모든 성좌 쌍 간의 가치관 거리 기반 친밀도 및 외교 상태 갱신
        """
        diplomacy_matrix = {}
        const_ids = list(self.tensor_field.constellations.keys())

        for i in range(len(const_ids)):
            for j in range(i + 1, len(const_ids)):
                id1, id2 = const_ids[i], const_ids[j]
                affinity, state = self.tensor_field.calculate_constellation_affinity(id1, id2)

                c1_name = self.tensor_field.constellations[id1].name
                c2_name = self.tensor_field.constellations[id2].name

                record = {
                    "affinity": round(affinity, 4),
                    "state": state.value,
                    "description": f"[{c1_name}] × [{c2_name}] 외교 상태: {state.value} (친밀도: {affinity:.2f})"
                }
                diplomacy_matrix[(id1, id2)] = record

        return diplomacy_matrix

    def execute_causal_trade(
        self,
        contract_id: str,
        initiator: ConstellationAgent,
        target: ConstellationAgent,
        amount: float,
        resource_desc: str
    ) -> Dict[str, Any]:
        """
        성좌 간 인과율 거래 체결
        (예: 지하 암시장 물류 보호를 대가로 [계약의 성좌]에게 인과력 20.0 지불)
        """
        if initiator.causal_power_pool < amount:
            return {
                "success": False,
                "reason": f"인과력 부족 (필요: {amount}, 보유: {initiator.causal_power_pool})"
            }

        initiator.causal_power_pool -= amount
        target.causal_power_pool += amount

        contract = TradeContract(
            contract_id=contract_id,
            initiator_id=initiator.constellation_id,
            target_id=target.constellation_id,
            causal_power_amount=amount,
            traded_resource_desc=resource_desc
        )
        self.active_contracts[contract_id] = contract

        msg = (
            f"⚡ [Divine Diplomacy] 성좌 '{initiator.name}'이(가) '{target.name}'에게 "
            f"인과력 {amount:.1f}를 지불하고 <{resource_desc}> 조약을 체결했습니다!"
        )
        self.chronicle_diplomacy_logs.append(msg)

        return {
            "success": True,
            "contract": contract,
            "log": msg
        }

    def trigger_schism_proxy_war(
        self,
        const_1: ConstellationAgent,
        const_2: ConstellationAgent,
        hero_1: HeroAlignmentState,
        hero_2: HeroAlignmentState
    ) -> Dict[str, Any]:
        """
        신좌 대리전(Schism Proxy War) 발동:
        가치관 반위상(Schism, S < -0.40) 상태인 두 성좌가 지상계 사도 영웅들에게 대리전 신탁 강제 부여.
        """
        affinity, state = self.tensor_field.calculate_constellation_affinity(
            const_1.constellation_id, const_2.constellation_id
        )

        if state != RelationState.SCHISM:
            return {
                "proxy_war_triggered": False,
                "reason": f"외교 상태가 Schism이 아닙니다 (현재 상태: {state.value})"
            }

        war_msg = (
            f"⚔️ [DIVINE PROXY WAR] 반위상 신좌 대리전 격돌! "
            f"성좌 '{const_1.name}'의 사도 '{hero_1.name}' vs "
            f"성좌 '{const_2.name}'의 사도 '{hero_2.name}'!"
        )
        self.chronicle_diplomacy_logs.append(war_msg)

        return {
            "proxy_war_triggered": True,
            "state": state.value,
            "constellation_1": const_1.name,
            "constellation_2": const_2.name,
            "apostle_1": hero_1.name,
            "apostle_2": hero_2.name,
            "log": war_msg
        }
