"""
continuous_gear_stat_system.py
===============================
변화 자체를 레이어화하여 고정기어(Fixed Gear)와 가변기어(Variable Gear) 형태의
동력 전달 메커니즘(Power Transmission Mechanism)으로 다루는 연속적 스탯 시스템.

기존 GAS(Gameplay Ability System)나 정적 스냅샷 수치 모델(+=, -=)의 단절된 상태값 대신,
입력 동력(Flux/Torque)이 고정 관성 축과 가변 변속 축의 연쇄를 지나며
실시간 유도(Derived)되는 연속적 동역학계(Continuous Dynamical System)를 구현합니다.
"""

from typing import Dict, Union, Optional
import math


class FixedGear:
    """
    ① 고정기어 (Fixed Gear: Structural Invariant Ratio)
    -------------------------------------------------
    - 역할: 시스템의 기본 관성(Momentum/Inertia)과 체급 항렬 구조를 결정합니다.
    - 특징: 레벨업 및 영구 성장은 수치를 직접 가산하는 것이 아니라,
           기어의 물리적 크기/이빨 수(Gear Ratio) 자체 및 관성 모멘트를 확장하는 행위입니다.
    - 효과: 동일한 외부 입력 동력이 들어왔을 때, 고정기어 비율($R_{fixed}$)이 크면
           기저 시스템에 전달되는 단위 에너지가 근본적으로 전환됩니다.
    """

    def __init__(
        self,
        base_level: int = 1,
        max_level: int = 100,
        growth_exponent: float = 1.2,
        base_inertia: float = 10.0
    ):
        self.current_level = base_level
        self.max_level = max_level
        self.growth_exponent = growth_exponent
        self.base_inertia = base_inertia

    def set_level(self, level: int) -> None:
        """레벨 설정 (기어의 관성 모멘트 및 고정기어비 재설정)."""
        self.current_level = max(1, min(self.max_level, level))

    def level_up(self, levels: int = 1) -> None:
        """레벨 상승으로 고정 기어 구조를 비가역적으로 확장."""
        self.set_level(self.current_level + levels)

    def get_fixed_ratio(self) -> float:
        """
        [고정 기어비 산출]
        R_fixed = (CurrentLevel / BaseLevel) ^ growth_exponent
        """
        return math.pow(float(self.current_level), self.growth_exponent)

    def get_moment_of_inertia(self) -> float:
        """
        [관성 모멘트]
        질량 및 관성 모멘트 I = base_inertia * (R_fixed)
        """
        return self.base_inertia * self.get_fixed_ratio()


class VariableGear:
    """
    ② 가변기어 (Variable Gear: Dynamic Impedance Matching)
    -----------------------------------------------------
    - 역할: 외부 상호작용 매개체(무기, 스킬, 버프/디버프)와 시스템 내부 사이의 변속비(CVT Ratio)를 조율합니다.
    - 특징:
      1) 무기 착용/해제: 동력 축에 물려있는 가변 기어 변속비 $R_{weapon}$을 조정합니다.
         무기 해제 시 $R \\to 0$으로 이행하며 클러치 분리(Clutch Disengage)가 일어납니다.
      2) 오버드라이브 (Buff): 가변 변속비가 Overdrive 영역($R_{buff} > 1.0$)으로 진입하여 출력을 증폭합니다.
      3) 마찰/슬립 (Debuff): 기어 축에 슬립(Slip)이나 구조적 마찰(Friction)이 발생하여 전달 효율이 저하됩니다.
    - 연속성: 변속비 $R_{var}(t)$는 목표 변속비(Target Ratio)로 부드럽게 이행(Shift Damping / S-Curve)합니다.
    """

    def __init__(
        self,
        shift_speed: float = 5.0,
        initial_ratio: float = 1.0,
        clutch_threshold: float = 0.01
    ):
        self.shift_speed = shift_speed            # 변속 이행 속도 (Hz / Damping Rate)
        self.current_ratio = initial_ratio        # 현재 유효 변속비
        self.weapon_gear_ratio = initial_ratio    # 무기/장비 기어비 저장

        # 오버드라이브(버프) 및 마찰/슬립(디버프) 계수
        self.overdrive_mult = 1.0                 # Overdrive Buff (> 1.0)
        self.friction_slip = 0.0                  # Friction Slip Debuff (0.0 ~ 1.0)

        # 클러치 상태 (Clutch Mechanism)
        self.clutch_threshold = clutch_threshold
        self.is_clutch_engaged = initial_ratio > clutch_threshold

        # 목표 변속비 초기 산출
        self.target_ratio = 1.0
        self._update_target_ratio()

    def equip_weapon(self, gear_ratio: float) -> None:
        """무기 장착: 목표 변속비를 변경하여 동력 전달축을 연결."""
        self.weapon_gear_ratio = max(0.0, gear_ratio)
        self._update_target_ratio()

    def unequip_weapon(self) -> None:
        """무기 해제: 변속비를 0으로 슬라이딩시켜 클러치를 분리(Clutch Disengage)."""
        self.weapon_gear_ratio = 0.0
        self._update_target_ratio()

    def set_overdrive_buff(self, multiplier: float) -> None:
        """오버드라이브 버프 설정 (Overdrive Region > 1.0)."""
        self.overdrive_mult = max(0.0, multiplier)
        self._update_target_ratio()

    def set_friction_debuff(self, friction: float) -> None:
        """마찰/슬립 디버프 설정 (Friction Drag Loss, 0.0 ~ 1.0)."""
        self.friction_slip = max(0.0, min(0.99, friction))

    def _update_target_ratio(self) -> None:
        """목표 총 가변 변속비 산출."""
        self.target_ratio = self.weapon_gear_ratio * self.overdrive_mult

    def tick(self, dt: float) -> None:
        """
        [연속적 변속 지연 (Smooth Shift Damping)]
        1프레임 불연속 도약 대신 물리적 관성에 맞춰 R_var(t)를 부드럽게 재정렬합니다.
        FInterpTo: R(t+dt) = R(t) + (R_target - R(t)) * (1 - e^(-shift_speed * dt))
        """
        if dt <= 0.0:
            return

        delta = self.target_ratio - self.current_ratio
        if abs(delta) > 1e-6:
            # exponential damping / smooth s-curve transition
            alpha = 1.0 - math.exp(-self.shift_speed * dt)
            self.current_ratio += delta * alpha
        else:
            self.current_ratio = self.target_ratio

        # 클러치 결합/분리 판정
        self.is_clutch_engaged = self.current_ratio > self.clutch_threshold

    def get_effective_ratio(self) -> float:
        """
        [효율이 반영된 유효 가변 기어비]
        R_var_effective = R_current * (1 - Friction_Slip)
        """
        if not self.is_clutch_engaged and self.current_ratio <= self.clutch_threshold:
            return 0.0

        efficiency = 1.0 - self.friction_slip
        return self.current_ratio * efficiency


class GearTransmissionSystem:
    """
    ③ 토크 구동축 (Torque Evaluator & Gear Transmission System)
    ---------------------------------------------------------
    - 역할: 입력 동력(Flux)과 [고정 기어비] × [가변 기어비] 연쇄를 통해
           최종 스탯(Output Kinetic Field)을 실시간 유도(Derived)합니다.
    - 핵심 아키텍처:
      1) 스탯 상태를 메모리에 가산/감산 방식(+=, -=)으로 저장하지 않습니다.
      2) 무기 교체, 버프, 디버프 시 수치 꼬임 버그가 원천 차단됩니다.
      3) 단일 스탯(Scalar) 및 다중 스탯 벡터(Vector/Dict)를 모두 지원합니다.
    """

    def __init__(
        self,
        base_flux: Union[float, Dict[str, float]] = 100.0,
        fixed_gear: Optional[FixedGear] = None,
        variable_gear: Optional[VariableGear] = None
    ):
        self.base_flux = base_flux
        self.fixed_gear = fixed_gear if fixed_gear is not None else FixedGear()
        self.variable_gear = variable_gear if variable_gear is not None else VariableGear()

    def tick(self, dt: float) -> None:
        """동역학계 시간 전진 (가변 기어 변속 보간 업데이트)."""
        self.variable_gear.tick(dt)

    def evaluate_output_torque(self) -> Union[float, Dict[str, float]]:
        r"""
        [동력 유도 연산기 (Torque Evaluator)]
        \tau_{out} = \tau_{in} \cdot R_{fixed} \cdot R_{var\_effective}

        스탯을 저장하지 않고, 필요한 순간 기어 축들의 곱으로 실시간 유도합니다.
        """
        r_fixed = self.fixed_gear.get_fixed_ratio()
        r_var = self.variable_gear.get_effective_ratio()
        total_transmission_ratio = r_fixed * r_var

        if isinstance(self.base_flux, (int, float)):
            return float(self.base_flux) * total_transmission_ratio
        elif isinstance(self.base_flux, dict):
            return {
                stat_key: float(val) * total_transmission_ratio
                for stat_key, val in self.base_flux.items()
            }
        else:
            raise TypeError(f"Unsupported base_flux type: {type(self.base_flux)}")

    def get_system_diagnostics(self) -> Dict[str, float]:
        """기계계 전반의 물리 상태 진단 정보 반환."""
        output = self.evaluate_output_torque()
        scalar_out = output if isinstance(output, float) else sum(output.values())

        return {
            "level": float(self.fixed_gear.current_level),
            "R_fixed": round(self.fixed_gear.get_fixed_ratio(), 4),
            "R_var_target": round(self.variable_gear.target_ratio, 4),
            "R_var_current": round(self.variable_gear.current_ratio, 4),
            "R_var_effective": round(self.variable_gear.get_effective_ratio(), 4),
            "friction_slip": round(self.variable_gear.friction_slip, 4),
            "overdrive_mult": round(self.variable_gear.overdrive_mult, 4),
            "clutch_engaged": float(self.variable_gear.is_clutch_engaged),
            "moment_of_inertia": round(self.fixed_gear.get_moment_of_inertia(), 4),
            "output_torque_total": round(scalar_out, 4)
        }
