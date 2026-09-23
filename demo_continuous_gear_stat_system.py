"""
demo_continuous_gear_stat_system.py
==================================
정적 스냅샷 수치 모델(GAS 등)과 연속적 기어 변속 동역학 모델(Continuous Gear Transmission)의
비교 시연 데모 스크립트.

이 데모는 두 모델 간의 다음 패러다임 차이를 구체적으로 보여줍니다:
1. 무기 교체 (Weapon Swap & Clutch Disengagement):
   - GAS 방식: 개별 Infinite Effect 제거 및 추가로 인한 즉각적/불연속적 단절(Spike)
   - 기어 방식: 변속비 $R_{var}(t)$가 슬라이딩되며 클러치가 분리/결합되는 부드러운 물리적 연속성
2. 레벨업 (Level Up / Growth):
   - GAS 방식: 정적 변수 가산 (`BaseAttack += 50`)
   - 기어 방식: 고정 기어 비율($R_{fixed}$) 및 관성 모멘트(Inertia)의 구조적 확장
3. 상태 이상 (Buff & Debuff):
   - GAS 방식: 개별 버프/디버프 이펙트 핸들 수동 관리 및 순서 꼬임 버그 위험
   - 기어 방식: 오버드라이브(Overdrive Region)와 마찰/슬립(Friction Slip)의 통일된 동력 전달 효율 연산
"""

import time
from core.physics.continuous_gear_stat_system import (
    FixedGear,
    VariableGear,
    GearTransmissionSystem
)


class ConventionalGASModel:
    """기존 정적 수치 모델 (GAS 방식 시뮬레이션)."""
    def __init__(self):
        self.base_attack = 100.0
        self.level = 1
        self.weapon_bonus = 0.0
        self.buff_multiplier = 1.0
        self.debuff_reduction = 0.0

    def get_final_attack(self) -> float:
        # 단절된 숫자를 메모리에 정적으로 기록하고 조합
        level_bonus = (self.level - 1) * 20.0
        net_base = self.base_attack + level_bonus + self.weapon_bonus
        return net_base * self.buff_multiplier * (1.0 - self.debuff_reduction)


def run_comparison_demo():
    print("=========================================================================")
    print("  CONTINUOUS GEAR TRANSMISSION SYSTEM vs CONVENTIONAL STATIC STAT MODEL  ")
    print("=========================================================================\n")

    # 1. 시스템 초기화
    gas_hero = ConventionalGASModel()

    fixed_gear = FixedGear(base_level=1, growth_exponent=1.2, base_inertia=10.0)
    variable_gear = VariableGear(shift_speed=4.0, initial_ratio=1.0)
    gear_hero = GearTransmissionSystem(
        base_flux={"attack": 100.0, "defense": 50.0},
        fixed_gear=fixed_gear,
        variable_gear=variable_gear
    )

    print("[Phase 1: Initial State (Level 1, Base Weapon Connected)]")
    variable_gear.equip_weapon(1.0)
    variable_gear.tick(1.0)  # Stabilize initial gear ratio

    gas_atk = gas_hero.get_final_attack()
    gear_diag = gear_hero.get_system_diagnostics()
    gear_atk = gear_hero.evaluate_output_torque()["attack"]

    print(f"  - [GAS Model] Final Attack: {gas_atk:.2f}")
    print(f"  - [Gear System] R_fixed: {gear_diag['R_fixed']:.2f} | R_var: {gear_diag['R_var_current']:.2f} | Final Attack: {gear_atk:.2f}")
    print("-" * 75 + "\n")

    # 2. 무기 교체 시연 (Smooth Transition & Clutch Disengage)
    print("[Phase 2: Weapon Swap (Unequip Heavy Sword -> Smooth Transition -> Equip Greatsword)]")
    print("  * GAS Model instantly jumps values across 1 frame.")
    print("  * Gear System slides CVT ratio R_var(t) with mechanical inertia & clutch disengagement.\n")

    # GAS 무기 교체
    gas_hero.weapon_bonus = 150.0  # Instant jump

    # Gear 무기 교체 (R_var = 2.5)
    variable_gear.equip_weapon(2.5)

    print(f"  {'Time(s)':<8} | {'GAS Attack':<12} | {'Gear R_var':<12} | {'Gear Attack':<12} | {'Clutch Engaged'}")
    print("  " + "-" * 65)

    dt = 0.1
    for t_step in range(8):
        current_time = t_step * dt
        gear_hero.tick(dt)
        g_atk = gas_hero.get_final_attack()
        diag = gear_hero.get_system_diagnostics()
        gr_atk = gear_hero.evaluate_output_torque()["attack"]
        clutch_str = "YES" if diag["clutch_engaged"] else "NO (Disengaged)"

        print(f"  {current_time:<8.1f} | {g_atk:<12.2f} | {diag['R_var_current']:<12.3f} | {gr_atk:<12.2f} | {clutch_str}")

    print("\n" + "-" * 75 + "\n")

    # 3. 레벨업 시연 (Fixed Ratio & Inertia Expansion)
    print("[Phase 3: Level Up (Growth / Structural Invariant Expansion)]")
    gas_hero.level = 10  # BaseAttack += level bonus
    fixed_gear.set_level(10)  # Gear ratio expansion & inertia momentum gain

    gas_atk = gas_hero.get_final_attack()
    diag = gear_hero.get_system_diagnostics()
    gear_atk = gear_hero.evaluate_output_torque()["attack"]

    print(f"  - [GAS Model] Level: {gas_hero.level} | Attack: {gas_atk:.2f}")
    print(f"  - [Gear System] Level: {fixed_gear.current_level} | R_fixed Ratio: {diag['R_fixed']:.2f} | Inertia: {diag['moment_of_inertia']:.2f} | Attack: {gear_atk:.2f}")
    print("-" * 75 + "\n")

    # 4. 버프 (Overdrive) & 디버프 (Friction Slip) 시연
    print("[Phase 4: Buffs & Debuffs (Overdrive Region & Friction Drag Loss)]")
    print("  * GAS Model adds multiple multiplier handles (* 1.5, * 0.8)")
    print("  * Gear System unifies all status effects into Power Transmission Efficiency.\n")

    # Overdrive Buff 1.5x & Friction Debuff 30%
    variable_gear.set_overdrive_buff(1.5)
    variable_gear.set_friction_debuff(0.3)
    variable_gear.tick(0.5)

    gas_hero.buff_multiplier = 1.5
    gas_hero.debuff_reduction = 0.3

    gas_atk = gas_hero.get_final_attack()
    diag = gear_hero.get_system_diagnostics()
    gear_out = gear_hero.evaluate_output_torque()

    print(f"  - [GAS Model] Final Attack: {gas_atk:.2f}")
    print(f"  - [Gear System] Overdrive: {diag['overdrive_mult']:.2f} | Friction Slip: {diag['friction_slip']:.2f} | Effective R_var: {diag['R_var_effective']:.2f}")
    print(f"  - [Gear System Derived Multi-Axis Output Vector]:")
    for stat_name, val in gear_out.items():
        print(f"      • {stat_name.capitalize()}: {val:.2f}")

    print("\n=========================================================================")
    print("  DEMO COMPLETE: Continuous Gear Transmission Architecture Verified.  ")
    print("=========================================================================")


if __name__ == "__main__":
    run_comparison_demo()
