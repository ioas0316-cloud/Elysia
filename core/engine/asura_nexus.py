"""
Elysia Core Engine: AsuraCheonmu Causal Nexus & Direct State Causality Module

This module implements the AsuraCheonmu Causal Nexus data structure along with zero-copy bitmask state encoding for high-performance direct state causality mapping.
"""

from dataclasses import dataclass, field
import struct
from typing import Any, Dict, List, Tuple


# Bitmask constants for Zero-Abstraction hardware signal dispatch
NEXUS_BIT_INACTIVE = 0x00
NEXUS_BIT_TRIGGERED = 0x01
NEXUS_BIT_DOMAIN_LOCKED = 0x02
NEXUS_BIT_PALETTE_INVERTED = 0x04
NEXUS_BIT_HIT_ACTIVE = 0x08
NEXUS_BIT_ANNIHILATE = 0x10


@dataclass
class AsuraCheonmuNexus:
    nexus_id: str = "SKILL_ASURA_CHEONMU"

    # 1. 원인 벡터 (Cause Vectors: 발동 조건)
    cause_vectors: dict = field(
        default_factory=lambda: {
            "required_weapon": "WEAPON_ASURA",  # 마검 아수라 장착 필수
            "min_soul_point": 100,             # 필살기 요구 SP
            "tp_cost": 150,                     # 행동력 소모
            "caster_state": "STATE_NORMAL",     # 발동 가능 상태
        }
    )

    # 2. 궤적 및 위상 전이 (Trajectory & Spatial Transition: 공간 연산)
    spatial_trajectory: dict = field(
        default_factory=lambda: {
            "domain_lock": True,                 # 발동 즉시 주변 타 노드의 시간(Δt) 0으로 동결
            "vector_paths": ["OCTA_DIRECTION_SLASH"],  # 8방향 교차 검기 궤적
            "teleport_sequence": [               # 16개 위치 잔상 궤적 좌표 (x, y, z)
                (0.0, 0.0, 0.0), (1.0, 2.0, 0.0), (3.0, 1.0, 0.0), (2.0, -1.0, 0.0),
                (-1.0, -2.0, 0.0), (-3.0, 0.0, 0.0), (-2.0, 2.0, 0.0), (0.0, 3.0, 0.0),
                (2.0, 3.0, 0.0), (4.0, 1.0, 0.0), (3.0, -2.0, 0.0), (1.0, -3.0, 0.0),
                (-2.0, -3.0, 0.0), (-4.0, -1.0, 0.0), (-3.0, 1.0, 0.0), (0.0, 0.0, 0.0),
            ],
        }
    )

    # 3. 인과 판정 (Causal Hit: 데미지 및 상태 결착)
    hit_causality: dict = field(
        default_factory=lambda: {
            "hit_count": 16,                      # 16연타 인과 연쇄
            "defense_bypass_ratio": 1.0,          # 방어력 100% 무시 절대 인과
            "target_scope": "AOE_SCREEN_ALL",     # 화면 내 모든 적대 노드
            "consequence_state": "STATE_ANNIHILATION",  # 타격 완료 시 즉사/절대 마비
        }
    )

    # 4. 현상 표현 (Visual Manifestation: 스프라이트/렌더링 매핑)
    visual_manifestation: dict = field(
        default_factory=lambda: {
            "color_inversion": "RED_BLACK_PALETTE",  # 화면 전체 적흑 반전
            "sprite_overlays": ["ASURA_BLADE_GLOW", "SCREEN_SLASH_CRACK"],
            "audio_trigger": "SFX_VOICE_ASURA_CHEONMU",
        }
    )

    def validate_activation(self, caster_sp: int, caster_tp: int, equipped_weapon: str, current_state: str) -> bool:
        """발동 조건(원인 벡터) 충족 여부 검증"""
        return (
            equipped_weapon == self.cause_vectors["required_weapon"]
            and caster_sp >= self.cause_vectors["min_soul_point"]
            and caster_tp >= self.cause_vectors["tp_cost"]
            and current_state == self.cause_vectors["caster_state"]
        )

    def to_flat_bit_array(self, state_flags: int) -> bytearray:
        """
        Zero-Copy / Zero-Abstraction을 위한 상태 데이터 팩킹.
        플랫 메모리 버퍼로 상태 비트 필드, hit_count, palette mode flag 등을 직렬화.
        """
        buffer = bytearray(64)
        # 0..7: 64-bit flag signal
        struct.pack_into("<Q", buffer, 0, state_flags)
        # 8..11: hit_count
        struct.pack_into("<I", buffer, 8, self.hit_causality["hit_count"])
        # 12..15: domain_lock (1 or 0)
        struct.pack_into("<I", buffer, 12, 1 if self.spatial_trajectory["domain_lock"] else 0)
        return buffer
