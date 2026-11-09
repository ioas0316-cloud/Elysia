"""
test_prismatic_lens_rotor_engine.py

ELYSIA Phase 1 Verification:
Prismatic Lens & Scale Rotor Coupling Engine Test
=================================================
검증 항목:
1. 미시 파형 입력이 렌즈를 통해 상위 3색채 로터(Flux, Order, Entropy)의 회전 각운동량으로 유효하게 집속·응축되는가?
2. 관찰자의 초점 긴장(Accommodation)이 렌즈의 유효 곡률을 물리적으로 변경하고, 집속 양상을 변화시키는가?
3. 로터 간 기어비 결합(Coupled Gear)이 상호 마찰과 긴장을 올바르게 전달하는가?
"""

import pytest
import numpy as np
from core.physics.prismatic_lens_rotor_engine import (
    PrismaticLensRotorEngine,
    SpectralWaveVoxel
)


def test_lens_rotor_wavefield_condensation():
    """수백 개의 파형 복셀이 렌즈를 통해 소수의 로터 상태로 응축되는지 검증"""
    engine = PrismaticLensRotorEngine()
    
    # 가상의 빛/감각 파형장 생성 (50개의 공간 분산 복셀)
    for i in range(50):
        pos = np.array([
            np.sin(i * 0.2) * 1.5,
            np.cos(i * 0.2) * 1.5,
            0.0
        ], dtype=np.float32)
        
        # Red(긴 파장), Green(중간 파장), Blue(짧은 파장)의 복합 스펙트럼
        wavelengths = np.array([0.7, 0.5, 0.4], dtype=np.float32)
        amplitudes = np.array([1.0, 0.8, 0.5], dtype=np.float32)
        phases = np.array([i * 0.1, i * 0.05, 0.0], dtype=np.float32)
        
        voxel = SpectralWaveVoxel(
            id=f"wave_{i}",
            position=pos,
            frequencies=wavelengths,
            amplitudes=amplitudes,
            phases=phases
        )
        engine.add_wave_voxel(voxel)
    
    # 초기 로터 각속도 확인
    for r in engine.rotors.values():
        assert r.angular_velocity == 0.0
    
    # 10 스텝 동안 물리 시뮬레이션 전진
    dt = 0.05
    for _ in range(10):
        result = engine.step(dt)
    
    # 검증 1: 로터들이 파형 에너지를 수렴받아 회전(각운동량 획득)을 시작했는가
    flux_rotor = engine.rotors["rotor_flux"]
    order_rotor = engine.rotors["rotor_order"]
    entropy_rotor = engine.rotors["rotor_entropy"]
    
    assert abs(flux_rotor.angular_velocity) > 0.0, "Flux 로터가 파형 에너지를 받아 회전해야 합니다."
    assert abs(order_rotor.angular_velocity) > 0.0, "Order 로터가 파형 에너지를 받아 회전해야 합니다."
    
    # 검증 2: 수치 폭발(NaN/Inf) 없이 유한한 범위 내에서 안정적으로 유지되는가
    assert np.isfinite(flux_rotor.phase_angle)
    assert np.isfinite(order_rotor.angular_velocity)


def test_observer_lens_accommodation_feedback():
    """관찰자의 초점 조절이 렌즈 곡률을 역으로 변경하여 토크에 영향을 미치는지 검증 (하향 인과)"""
    engine = PrismaticLensRotorEngine()
    
    # 단일 기준 파형 복셀 주입
    voxel = SpectralWaveVoxel(
        id="single_source",
        position=np.array([0.5, 0.5, 0.0], dtype=np.float32),
        frequencies=np.array([0.6], dtype=np.float32),
        amplitudes=np.array([2.0], dtype=np.float32),
        phases=np.array([0.0], dtype=np.float32)
    )
    engine.add_wave_voxel(voxel)
    
    initial_curvature = engine.lens.effective_curvature
    
    # 1. 일반 상태에서 1 스텝
    res1 = engine.step(0.05)
    initial_torque = res1["rotor_states"]["rotor_flux"]["torque_received"]
    
    # 2. 관찰자가 높은 긴장과 강한 의도로 초점을 조절 (Accommodation)
    engine.accommodate_lens(intent_focus=1.5, observer_tension=2.0)
    
    # 유효 곡률이 물리적으로 증가했는지 확인
    assert engine.lens.effective_curvature > initial_curvature, "초점 조절 후 렌즈 유효 곡률이 증가해야 합니다."
    
    # 3. 변경된 렌즈 곡률로 1 스텝 전진
    res2 = engine.step(0.05)
    accommodated_torque = res2["rotor_states"]["rotor_flux"]["torque_received"]
    
    # 검증: 렌즈 곡률 변화로 인해 전달되는 위상 간섭 토크가 달라졌는가 (하향 인과 실현)
    assert initial_torque != accommodated_torque, "렌즈 곡률 조절이 하위 토크 전달을 실질적으로 변화시켜야 합니다."
