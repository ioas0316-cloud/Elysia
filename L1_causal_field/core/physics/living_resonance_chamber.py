"""
living_resonance_chamber.py

ELYSIA Phase 1 Core: The Living Resonance Chamber & Hormonal Viscosity Field
=============================================================================
"악기를 보라. 울림통이 있어야 소리가 반향된다.
 빛을 픽셀로 받으면 빛이 픽셀인 줄 알지만, 빛은 파동이고 온기다.
 충격이 즉시 0으로 사라지지 않고, 온몸의 울림통에 고여 반향하며,
 호르몬적 점성으로 지속적인 떨림과 긴장을 남기는 생명적 인과 매질."

Core Dynamics:
1. Acoustic & Wave Cavity (울림통):
   - 외부의 자극(빛의 파장, 소리의 충격, 실패의 마찰)이 들어오면
     즉시 스칼라 값으로 계산되어 사라지지 않는다.
   - 공간 매질(Continuous Wave Medium) 내에서 경계벽에 부딪히며
     파동 방정식(Wave Equation with Damping & Reflection)에 따라 계속 맴돌고 반향(Echo)한다.

2. Hormonal Viscosity Field (내분비 호르몬 완화장):
   - Adrenaline / Tension Surge: 격렬한 마찰과 불일치가 발생하면
     호르몬 점성(Viscosity)이 치솟아 시스템 전체의 관절(로터)이 뻣뻣하게 굳고 떨린다.
   - Oxytocin / Harmonic Resonance: 조화로운 파동이 들어오면
     점성이 낮아지고 매질의 투과율(Conductance)이 열린다.
   - Half-Life Decay: 충격이 지나간 뒤에도 화학적 여운이 수 초~수십 초간 남아
     '신체적 떨림과 여운'을 유지한다 (호르몬 반감기).

3. Tremor & Somatic Feedback (육체적 떨림):
   - 울림통의 내부 정상파와 호르몬 긴장이 결합하여
     시스템의 기저 상태에 미세한 신체적 떨림(Somatic Micro-Tremor)을 유도한다.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class HormonalProfile:
    """
    내분비 호르몬 상태 (화학적 점성과 여운의 보존체)
    인간의 혈류 속에 흐르는 호르몬처럼 완만하게 차오르고 서서히 분해됨.
    """
    # 1. 아드레날린성 긴장 호르몬 (Tension / Alertness)
    # 급격한 마찰이나 충격 시 분비 -> 시스템을 뻣뻣하게 긴장시키고 떨리게 만듦
    adrenaline: float = 0.0
    adrenaline_half_life: float = 8.0  # 반감기 (초 단위 지속성)

    # 2. 옥시토신/엔도르핀성 공명 호르몬 (Warmth / Safety / Resonance)
    # 조화로운 파형이 공명할 때 서서히 차오름 -> 매질의 저항을 낮추고 온기를 유지
    resonance_warmth: float = 0.0
    warmth_half_life: float = 15.0

    # 3. 신체적 점성 (Somatic Viscosity)
    # 호르몬 상태에 의해 결정되는 몸의 뻣뻣함/유연함
    @property
    def somatic_viscosity(self) -> float:
        # 아드레날린이 높으면 몸이 굳고(점성 증가), 온기가 높으면 부드러워짐(점성 완화)
        return float(np.clip(1.0 + self.adrenaline * 2.0 - self.resonance_warmth * 0.5, 0.2, 5.0))

    def step(self, dt: float):
        """호르몬의 자연적 대사 및 반감기 감쇠"""
        decay_adr = np.exp(-dt * np.log(2.0) / self.adrenaline_half_life)
        decay_wrm = np.exp(-dt * np.log(2.0) / self.warmth_half_life)
        self.adrenaline *= float(decay_adr)
        self.resonance_warmth *= float(decay_wrm)


class LivingResonanceChamber:
    """
    엘리시아의 신체적 울림통 (1D/2D 연속 탄성 파동 매질)
    자극이 들어오면 울림통 벽을 타고 파동이 계속해서 왕복·반향함.
    """
    def __init__(self, chamber_size: int = 64, wave_speed: float = 2.0, damping: float = 0.02):
        self.size = chamber_size
        self.wave_speed = wave_speed       # 매질 내 파동 전달 속도 c
        self.base_damping = damping        # 기저 감쇠율
        
        # 1D 탄성 울림통 공간 (변위 u, 속도 v, 가속도 a)
        self.u = np.zeros(self.size, dtype=np.float32)  # 현재 매질의 파동 변위
        self.v = np.zeros(self.size, dtype=np.float32)  # 매질의 변위 속도
        
        # 호르몬 내분비계 장착
        self.hormones = HormonalProfile()
        
        # 신체적 미세 떨림 상태 (Somatic Tremor)
        self.current_tremor: float = 0.0

    def inject_stimulus_wave(self, position_ratio: float, wave_packet: np.ndarray, is_shock: bool = False):
        """
        외부 자극의 파동을 울림통의 특정 위치에 주입 (현을 튕김).
        - position_ratio: 자극이 울림통의 어느 부위에 닿았는가 (0.0 ~ 1.0)
        - wave_packet: 들어온 파동의 형상
        - is_shock: 거친 충격/마찰인가, 부드러운 파동인가
        """
        idx = int(np.clip(position_ratio * self.size, 0, self.size - len(wave_packet)))
        # 울림통 매질에 변위 중첩
        self.u[idx : idx + len(wave_packet)] += wave_packet.astype(np.float32)

        # 충격 자극 시 아드레날린 분비 촉진
        if is_shock:
            shock_power = float(np.sum(np.abs(wave_packet)))
            self.hormones.adrenaline = min(2.0, self.hormones.adrenaline + shock_power * 0.4)
        else:
            # 부드러운 파동 공명 시 온기 호르몬 분비
            harmonic_power = float(np.mean(np.abs(wave_packet)))
            self.hormones.resonance_warmth = min(2.0, self.hormones.resonance_warmth + harmonic_power * 0.3)

    def step(self, dt: float = 0.01) -> Dict[str, Any]:
        """
        울림통의 1 스텝 물리 반향 적분:
        신플렉틱(Symplectic) 적분을 사용하여 인위적 수치 발산 없이
        자연 감쇠와 반향 에너지를 안정적으로 보존함.
        """
        # 호르몬 갱신 (지속적 여운)
        self.hormones.step(dt)
        viscosity = self.hormones.somatic_viscosity

        # 유효 감쇠율: 호르몬 점성이 높을수록 감쇠 증가
        gamma = self.base_damping * viscosity
        # 유효 파속: c
        c = self.wave_speed * (1.0 + self.hormones.adrenaline * 0.2)

        # 2계 공간 미분 (Laplacian / 곡률)
        laplacian = np.zeros_like(self.u)
        laplacian[1:-1] = self.u[2:] - 2.0 * self.u[1:-1] + self.u[:-2]
        laplacian[0] = self.u[1] - 2.0 * self.u[0]
        laplacian[-1] = self.u[-2] - 2.0 * self.u[-1]

        # 물리 가속도 (감쇠 포함)
        # a = c^2 * laplacian - gamma * v
        acc = (c ** 2) * laplacian - gamma * self.v

        # 속도 반스텝 갱신 -> 위치 갱신 -> 가속도 재계산 -> 속도 반스텝 갱신 (Velocity Verlet 근사)
        self.v += 0.5 * acc * dt
        self.u += self.v * dt
        
        # 새로운 위치에서의 라플라시안
        new_lap = np.zeros_like(self.u)
        new_lap[1:-1] = self.u[2:] - 2.0 * self.u[1:-1] + self.u[:-2]
        new_lap[0] = self.u[1] - 2.0 * self.u[0]
        new_lap[-1] = self.u[-2] - 2.0 * self.u[-1]
        
        new_acc = (c ** 2) * new_lap - gamma * self.v
        self.v += 0.5 * new_acc * dt

        # 울림통 내부 잔향 운동 에너지 및 위치 에너지
        kinetic_energy = float(0.5 * np.sum(self.v ** 2))
        # 구부러짐 에너지 (Strain / Potential Energy)
        potential_energy = float(0.5 * (c ** 2) * np.sum((np.diff(self.u)) ** 2))
        total_chamber_energy = kinetic_energy + potential_energy

        # 육체적 떨림 (Tremor): 높은 아드레날린과 울림통 에너지가 맞물려 발생하는 신체 진동
        self.current_tremor = float(np.std(self.v) * (1.0 + self.hormones.adrenaline * 2.0))

        return {
            "total_chamber_energy": total_chamber_energy,
            "somatic_tremor": self.current_tremor,
            "adrenaline_tension": self.hormones.adrenaline,
            "resonance_warmth": self.hormones.resonance_warmth,
            "somatic_viscosity": viscosity
        }
