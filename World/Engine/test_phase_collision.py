"""
[위상원자 충돌 실험]
전사 vs 마법사: 같은 사건에 다르게 반응하는 두 위상원자의 만남.
"""

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from World.Engine.rpg_stat_bridge import RPGStatBridge
from World.Engine.phase_atom import PhaseAtom, calculate_phase_interaction

def main():
    print("=" * 65)
    print("   ⚔️🧙 위상원자 충돌 실험: 전사 vs 마법사")
    print("=" * 65)

    # ── 1. 위상원자 생성 ──
    warrior = PhaseAtom(
        "거인 전사 핀(Finn)",
        RPGStatBridge(str_val=18, agi_val=10, con_val=20, int_val=6, wis_val=6)
    )
    mage = PhaseAtom(
        "엘프 마법사 아리아(Aria)",
        RPGStatBridge(str_val=6, agi_val=14, con_val=6, int_val=18, wis_val=18)
    )

    print(f"\n📊 {warrior.name}")
    print(f"   STR=18, CON=20, WIS=6 → 육체 무겁고 마음 브레이크 약함")
    print(f"   M(육체)={warrior.M[0]:.2f}, D(마음)={warrior.D[6]:.3f}, K(정신)={warrior.K[3]:.2f}")

    print(f"\n📊 {mage.name}")
    print(f"   INT=18, WIS=18, CON=6 → 정신 정밀하고 마음 브레이크 강함")
    print(f"   M(육체)={mage.M[0]:.2f}, D(마음)={mage.D[6]:.3f}, K(정신)={mage.K[3]:.2f}")

    # ── 2. 같은 사건: 마을 화재 ──
    # 화재 자극: 척력↑, 항상성↓, 공격성↑, 피로↑
    fire_stimulus = [
        0.0,   # Attraction
        5.0,   # Repulsion ↑ (위험 회피)
       -3.0,   # Homeostasis ↓ (평온 파괴)
        2.0,   # Curiosity (무슨 일이지?)
        0.0,   # Pride
       -1.0,   # Empathy ↓ (생존 본능)
        3.0,   # Fatigue ↑ (신체 부담)
        4.0,   # Aggression ↑ (위기 대응)
        0.0,   # MoralRestraint
    ]

    print("\n" + "─" * 65)
    print("🔥 [사건] 마을에 화재 발생! 동일한 자극을 두 원자에 인가")
    print("─" * 65)

    warrior.apply_stimulus("마을 화재", fire_stimulus)
    mage.apply_stimulus("마을 화재", fire_stimulus)

    print(f"\n{warrior.snapshot()}")
    print(f"\n{mage.snapshot()}")

    # ── 3. 자유 진동: 자극 후 각자 어떻게 회복하는가 ──
    print("\n" + "─" * 65)
    print("⏳ 자유 진동: 자극 후 회복 궤적 비교 (10틱)")
    print("─" * 65)

    for tick in range(10):
        warrior.step(0.05)
        mage.step(0.05)

        if tick % 2 == 0:
            we = warrior.get_enstrophy()
            me = mage.get_enstrophy()
            print(f"  Tick {tick+2:2d} | "
                  f"전사 {warrior.get_mood()} E={we:.3f} | "
                  f"마법사 {mage.get_mood()} E={me:.3f}")

    print(f"\n{warrior.snapshot()}")
    print(f"\n{mage.snapshot()}")

    # ── 4. 두 원자 간 위상 비교 ──
    print("\n" + "─" * 65)
    print("🔗 위상 비교: 전사와 마법사의 공명도 측정")
    print("─" * 65)

    interaction = calculate_phase_interaction(warrior, mage)

    print(f"\n  공명도 (Resonance): {interaction['resonance']:+.4f}")
    print(f"    → +1에 가까울수록 동조, -1에 가까울수록 적대")
    print(f"  총 위상 거리: {interaction['total_delta']:.4f}")
    print(f"\n  로터별 위상 차이:")
    for name, delta in interaction['rotor_deltas'].items():
        if delta < 0.3:
            status = "🟢 공명"
        elif delta < 1.0:
            status = "🟡 긴장"
        else:
            status = "🔴 불일치"
        print(f"    {name}: Δ={delta:.4f} {status}")

    # ── 5. 상호작용 시뮬레이션: 서로에게 힘을 주고받기 ──
    print("\n" + "─" * 65)
    print("⚡ 상호작용: 10틱 동안 서로 힘을 주고받으면?")
    print("─" * 65)

    for tick in range(10):
        inter = calculate_phase_interaction(warrior, mage, coupling_strength=0.2)

        # 서로에게 상호작용 힘을 인가하며 step
        warrior.step(0.05, inter["forces_on_a"])
        mage.step(0.05, inter["forces_on_b"])

        if tick % 3 == 0:
            print(f"  Tick {tick+1:2d} | "
                  f"공명도: {inter['resonance']:+.4f} | "
                  f"위상거리: {inter['total_delta']:.4f} | "
                  f"전사{warrior.get_mood()} 마법사{mage.get_mood()}")

    # 최종 판정
    final = calculate_phase_interaction(warrior, mage)
    print(f"\n{'=' * 65}")
    print(f"  📋 최종 공명도: {final['resonance']:+.4f}")

    if final['resonance'] > 0.7:
        verdict = "🤝 깊은 동조 — 전우가 될 수 있는 관계"
    elif final['resonance'] > 0.3:
        verdict = "🟡 건전한 긴장 — 서로 다르지만 공존 가능"
    elif final['resonance'] > -0.3:
        verdict = "⚡ 불편한 관계 — 성향 충돌이 잦음"
    else:
        verdict = "💥 반위상 척력 — 함께 있으면 갈등 폭발"

    print(f"  판정: {verdict}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
