"""
revelation_auction.py
=====================
Elysia Causal Engine - Revelation Auction & Hero Apostasy Mechanics
Handles competitive bidding among constellations when a hero faces critical ordeals.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from modules.causal_game_engine.alignment_field import (
    AlignmentVector,
    HeroAlignmentState,
    AlignmentTensorField
)
from modules.causal_game_engine.npc_constellation_engine import ConstellationAgent


@dataclass
class BidProposal:
    """성좌의 신탁 입찰 제안서"""
    constellation_id: str
    constellation_name: str
    causal_power_bid: float  # 투입 인과력 수치
    revelation_text: str  # 신탁 문구
    offered_artifact_name: str  # 은밀히 제안하는 아티팩트/보구
    alignment_resonance: float  # 가치관 공명도 [0.0 ~ 1.0]
    total_attraction_score: float = 0.0  # 총 유혹 수치


class RevelationAuctionHouse:
    """
    신탁 경매 및 영웅 배교(Apostasy) 경매장.
    영웅이 시련(Ordeal)이나 사경(Critical HP / Famine)에 도달했을 때
    플레이어 및 NPC 성좌들의 입찰 유혹과 영웅의 자율 가치관 연산을 거쳐 영웅의 수용/배교를 결정.
    """

    def __init__(self, tensor_field: AlignmentTensorField):
        self.tensor_field = tensor_field

    def trigger_ordeal_auction(
        self,
        hero: HeroAlignmentState,
        ordeal_type: str,
        bids: List[BidProposal]
    ) -> Dict[str, Any]:
        """
        시련 발생 시 신탁 경매 진행 및 영웅의 수용 연산.
        영웅의 5대 스탯/가치관 벡터와 성좌가 제시한 조건의 공명도를 바탕으로 입찰 승자 산출.
        """
        if not bids:
            return {
                "auction_status": "NO_BIDS",
                "winning_bid": None,
                "is_apostasy": False
            }

        winning_bid = None
        max_score = -9999.0

        for bid in bids:
            # 총 유혹 수치 = (인과력 투입량 * 0.5) + (가치관 공명도 * 100.0) + (100 - Hero.SPI) * 0.3
            # SPI(정신력)가 낮을수록 낯선 성좌의 파격적 입찰에 유혹당하기 쉬움
            const_node = self.tensor_field.constellations.get(bid.constellation_id)
            if not const_node:
                continue

            dist = hero.current_alignment.distance_to(const_node.alignment)
            resonance = max(0.0, 1.0 - (dist / 2.828))
            bid.alignment_resonance = float(resonance)

            score = (
                (bid.causal_power_bid * 0.6) +
                (resonance * 80.0) +
                ((100.0 - hero.spi_stat) * 0.4)
            )
            bid.total_attraction_score = float(score)

            if score > max_score:
                max_score = score
                winning_bid = bid

        # 배교 여부 판단: 승리한 성좌가 플레이어가 아닌 NPC 성좌이고,
        # 이전 주 후원 성좌와 다를 경우 배교(Apostasy) 발생
        is_apostasy = False
        if winning_bid and winning_bid.constellation_id != hero.bound_constellation_id:
            if hero.bound_constellation_id is not None:
                is_apostasy = True

            # 영웅의 주 후원 성좌 업데이트 및 가치관 표류 충격 적용
            prev_bound = hero.bound_constellation_id
            hero.bound_constellation_id = winning_bid.constellation_id

            const_win = self.tensor_field.constellations[winning_bid.constellation_id]
            shock_vec = AlignmentVector(
                x=(const_win.alignment.x - hero.current_alignment.x) * 0.5,
                y=(const_win.alignment.y - hero.current_alignment.y) * 0.5
            )
            self.tensor_field.update_hero_alignment_drift(
                hero.hero_id, event_shock_vector=shock_vec
            )

        return {
            "auction_status": "COMPLETED",
            "ordeal_type": ordeal_type,
            "hero_id": hero.hero_id,
            "hero_name": hero.name,
            "winning_bid": winning_bid,
            "is_apostasy": is_apostasy,
            "description": (
                f"[{ordeal_type}] 시련 중 '{hero.name}'이(가) "
                f"'{winning_bid.constellation_name}'의 신탁('{winning_bid.revelation_text}')을 수용했습니다!"
                if winning_bid else "수용된 신탁이 없습니다."
            )
        }
