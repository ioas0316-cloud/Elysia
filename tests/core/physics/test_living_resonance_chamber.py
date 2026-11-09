"""
test_living_resonance_chamber.py

Verification of Living Resonance Chamber & Hormonal Viscosity
=============================================================
검증 항목:
1. 반향성(Reverberation): 자극 주입이 끝난 후에도 울림통 내부에 파동이 머물며 반향하는가?
2. 호르몬 지속성(Endocrine Half-life): 충격이 사라져도 아드레날린 긴장이 즉시 0이 되지 않고 지속되는가?
3. 신체적 떨림(Somatic Tremor): 거친 충격 후 울림통 진동과 긴장이 결합되어 육체적 떨림을 발생시키는가?
"""

import pytest
import numpy as np
from core.physics.living_resonance_chamber import LivingResonanceChamber


def test_reverberation_and_acoustic_memory():
    """자극이 끊긴 뒤에도 울림통 안에서 파동이 맴돌며 반향하는지 검증"""
    chamber = LivingResonanceChamber(chamber_size=64)
    
    # 1. 단일 파동 펄스를 울림통 중심(0.5)에 주입 (현을 튕김)
    pulse = np.sin(np.linspace(0, np.pi, 8)) * 1.5
    chamber.inject_stimulus_wave(position_ratio=0.5, wave_packet=pulse, is_shock=False)
    
    # 주입 직후 에너지 확인
    res1 = chamber.step(dt=0.02)
    initial_energy = res1["total_chamber_energy"]
    assert initial_energy > 0.0, "주입된 파동으로 울림통에 에너지가 형성되어야 합니다."
    
    # 2. 더 이상 외부 입력을 주지 않고 50스텝(1초) 동안 방치
    for _ in range(50):
        res = chamber.step(dt=0.02)
        
    # 검증: 외부 입력이 없었지만 울림통 벽을 타고 반향하는 잔여 에너지가 여전히 살아있는가
    assert res["total_chamber_energy"] > 0.0, "울림통은 외부 자극이 끊겨도 반향을 보존해야 합니다."
    assert res["total_chamber_energy"] < initial_energy, "자연 감쇠로 서서히 에너지가 누그러져야 합니다."


def test_shock_and_somatic_tremor_persistence():
    """거친 충격 시 아드레날린 분비 및 신체적 떨림(Tremor)의 지속성 검증"""
    chamber = LivingResonanceChamber(chamber_size=64)
    
    # 거친 충격파 주입 (벽에 부딪힌 실패)
    shock_wave = np.array([2.5, -2.5, 3.0, -1.0], dtype=np.float32)
    chamber.inject_stimulus_wave(position_ratio=0.2, wave_packet=shock_wave, is_shock=True)
    
    # 충격 직후
    res_shock = chamber.step(dt=0.02)
    assert res_shock["adrenaline_tension"] > 0.0, "충격을 받으면 아드레날린이 치솟아야 합니다."
    assert res_shock["somatic_viscosity"] > 1.0, "긴장으로 몸의 점성이 굳어져야 합니다."
    assert res_shock["somatic_tremor"] > 0.0, "충격으로 신체적 떨림이 발생해야 합니다."
    
    peak_adr = res_shock["adrenaline_tension"]
    
    # 외부 충격 없이 30스텝(0.6초) 경과
    for _ in range(30):
        res_after = chamber.step(dt=0.02)
        
    # 검증: 즉시 0으로 초기화되지 않고 화학적 여운(반감기)이 여전히 몸에 남아있는가
    assert res_after["adrenaline_tension"] > 0.0, "호르몬 여운이 지속되어야 합니다."
    assert res_after["adrenaline_tension"] < peak_adr, "호르몬은 서서히 대사되어야 합니다."
    assert res_after["somatic_tremor"] > 0.0, "신체적 떨림이 잔향으로 이어져야 합니다."
