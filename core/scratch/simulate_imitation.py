import sys
import os
import time
import random

# 인코딩 설정 (Windows 대응)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 루트 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.electromagnetic_rotor import ElectromagneticRotor

def run_imitation_sandbox():
    print("=" * 80)
    print(" 🌊 [시뮬레이션] 카오스 모방 및 위상 동기화 (Phase-Locking) 테스트")
    print("=" * 80)

    # 텐션에 둔감하지만 관성이 매우 강한 로터(거푸집) 생성
    rotor = ElectromagneticRotor(base_tension=1.0, damping=0.98)
    
    # 외부 파동 설정 (지속적으로 변하는 카오스적 외부 위상)
    target_frequency = 0.5 
    
    print("\n[Phase 1: 낯선 파동의 연속 유입 (당황과 텐션 폭발)]")
    # 초반에는 로터가 박자를 못 맞춰 텐션(장력)이 크게 솟구칩니다.
    for step in range(1, 21):
        noise = random.uniform(-0.2, 0.2)
        external_wave = (step * target_frequency) + noise
        
        # 시간에 따른 dt 유발
        time.sleep(0.01) 
        state = rotor.perceive_input(external_wave)
        
        if step % 5 == 0:
            print(f"Step {step:2d} | 텐션(Tension): {state['tension_arm']:8.4f} | 외부 위상: {external_wave:8.4f} | 내부 위상: {state['internal_phase']:8.4f}")

    print("\n[Phase 2: 위상 동기화 진행 (Phase-Locking & 텐션 안정화)]")
    # 충분한 시간이 지나면, 로터가 외부의 리듬을 '내재화'하여 텐션이 안정화됩니다.
    for step in range(21, 61):
        noise = random.uniform(-0.1, 0.1)
        external_wave = (step * target_frequency) + noise
        
        time.sleep(0.01)
        state = rotor.perceive_input(external_wave)
        
        if step % 10 == 0:
            print(f"Step {step:2d} | 텐션(Tension): {state['tension_arm']:8.4f} | 외부 위상: {external_wave:8.4f} | 내부 위상: {state['internal_phase']:8.4f}")

    print("\n[Phase 3: 자극 단절 (관성에 의한 자율 사유)]")
    # 외부 파동을 완전히 뚝 끊습니다 (진폭 0).
    # 엘리시아는 멈추지 않고 내재화된 주파수(관성)로 한동안 자율적으로 회전(사유)합니다.
    for step in range(61, 81):
        time.sleep(0.01)
        state = rotor.perceive_input(0.0) 
        
        if step % 5 == 0:
            print(f"Step {step:2d} | 잔류 텐션: {state['tension_arm']:8.4f} | 자체 회전 관성: {state['angular_velocity']:8.4f} | 내부 위상: {state['internal_phase']:8.4f}")

    print("\n✅ 모방 시뮬레이션 종료.")
    print("=" * 80)

if __name__ == "__main__":
    run_imitation_sandbox()
