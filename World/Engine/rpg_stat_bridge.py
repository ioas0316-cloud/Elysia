"""
[RPG STAT BRIDGE - PHYSICAL STAT CONVERTER]
============================================
World.Engine.rpg_stat_bridge

"Translating classic RPG stats (STR, AGI, CON, INT, WIS) into
 the universal oscillator equation: M·ẍ + D·ẋ + K·x = F"

Canonical location: World/Engine/rpg_stat_bridge.py
Design reference: docs/ETERNOS_CODEX/20_ROTOR_SCALE_KINGDOM_ARCHITECTURE.md §3

Mapping:
    CON (체력) → Mass M        : 충격을 견디는 묵직함
    WIS (지혜) → Damping D     : 카오스를 진정시키는 브레이크
    INT (지능) → Stiffness K   : 목표 상태로의 정밀한 복원력
    STR (힘)   → Force mult F  : 세상에 개입하는 원동력
    AGI (민첩) → Speed factor v : 변화에 반응하는 순발력
"""

import math
from typing import Dict, List

# Pure Python — numpy 의존성 제거 (Core.Keystone.sovereign_math 정신 계승)

class RPGStatBridge:
    """5대 RPG 스탯을 로터 물리 매개변수(M, D, K)로 환원한다."""

    def __init__(self, str_val: int = 10, agi_val: int = 10,
                 con_val: int = 10, int_val: int = 10, wis_val: int = 10):
        self.stats = {
            "STR": str_val,  # Strength  (힘)
            "AGI": agi_val,  # Agility   (민첩)
            "CON": con_val,  # Constitution (체력)
            "INT": int_val,  # Intelligence (지능)
            "WIS": wis_val,  # Wisdom    (지혜)
        }

    def convert_to_physical_params(self, dims: int = 3) -> dict:
        """
        스탯 → 물리 상수 변환.

        Returns:
            M (list[float]): 질량 관성 벡터 (dims 차원)
            D (list[float]): 감쇠 계수 벡터
            K (list[float]): 결합 강성 벡터
            speed_factor (float): AGI 기반 시간 스케일러
            force_multiplier (float): STR 기반 외력 배율
        """
        # CON → Mass M : 0.5 (CON=1) ~ 3.0 (CON=50)
        mass = 0.5 + (self.stats["CON"] * 0.05)
        M = [mass] * dims

        # WIS → Damping D : 0.05 (WIS=1) ~ 0.8 (WIS=50)
        damping = 0.05 + (self.stats["WIS"] * 0.015)
        D = [damping] * dims

        # INT → Stiffness K : 0.5 (INT=1) ~ 3.0 (INT=50)
        stiffness = 0.5 + (self.stats["INT"] * 0.05)
        K = [stiffness] * dims

        # AGI → 반응 속도 배율
        speed_factor = 1.0 + (self.stats["AGI"] * 0.05)

        # STR → 외력 출력 배율
        force_multiplier = 1.0 + (self.stats["STR"] * 0.08)

        return {
            "M": M, "D": D, "K": K,
            "speed_factor": speed_factor,
            "force_multiplier": force_multiplier,
        }

    def __repr__(self):
        s = self.stats
        return (f"RPGStatBridge(STR={s['STR']}, AGI={s['AGI']}, "
                f"CON={s['CON']}, INT={s['INT']}, WIS={s['WIS']})")


# ──────────────────────────────────────────────
# Self-test: 거인 전사 vs 엘프 마법사 충격 시뮬레이션
# ──────────────────────────────────────────────
def _simulate(profile_name: str, params: dict, dims: int = 3):
    """순수 파이썬 2차 운동방정식 시뮬레이션."""
    x = [0.0] * dims
    v = [0.0] * dims
    dt = 0.05 * params["speed_factor"]

    # 초기 충격: 속도에 임펄스 인가
    v[0] = 8.0 / params["M"][0]

    print(f"\n🎬 [{profile_name}] 외력 충격 시뮬레이션:")
    for step in range(6):
        for i in range(dims):
            a = (-params["D"][i] * v[i] - params["K"][i] * x[i]) / params["M"][i]
            v[i] += a * dt
            x[i] += v[i] * dt

        instability = sum(vi ** 2 for vi in v)
        if instability < 0.1:
            mood = "🧘 [평온] 항상성 안착"
        elif instability < 2.0:
            mood = "🟡 [동요] 평정 회복 중"
        else:
            mood = "🔥 [폭주] 혼란/흥분 상태"

        print(f"  Step {step+1} | 위상: {x[0]:.3f} rad | "
              f"카오스: {instability:.3f} | {mood}")


if __name__ == "__main__":
    print("=" * 60)
    print("   🛡️ RPG 5대 스탯 → 로터 물리 거동 비교 시뮬레이션")
    print("=" * 60)

    warrior = RPGStatBridge(str_val=18, agi_val=10, con_val=20, int_val=6, wis_val=6)
    mage = RPGStatBridge(str_val=6, agi_val=12, con_val=6, int_val=18, wis_val=18)

    pw = warrior.convert_to_physical_params(3)
    pm = mage.convert_to_physical_params(3)

    print(f"\n📊 {warrior} → M={pw['M'][0]:.2f}, D={pw['D'][0]:.3f}, K={pw['K'][0]:.2f}")
    print(f"📊 {mage} → M={pm['M'][0]:.2f}, D={pm['D'][0]:.3f}, K={pm['K'][0]:.2f}")

    _simulate("거인 전사", pw)
    _simulate("엘프 마법사", pm)
    print("=" * 60)
