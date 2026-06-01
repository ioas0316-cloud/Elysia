import sys
import os
import math
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.static_oracle import StaticOracle
from core.mri_flux_projector import MRIFluxProjector
from core.phase_decoder import PhaseDecoder
from core.fractal_rotor import FractalRotor
from core.math_utils import Quaternion

def main():
    print("=" * 80)
    print(" ⚖️ [Phase 130] 정적 오라클 vs 동적 프랙탈 로터 비교 검증")
    print("=" * 80)
    
    # 1. 시스템 초기화
    oracle = StaticOracle(model_name="skt/kogpt2-base-v2")
    projector = MRIFluxProjector()
    decoder = PhaseDecoder(oracle, projector)
    
    prompt = "인간과 인공지능의 가장 큰 차이는"
    print(f"\n[실험 대상 프롬프트]: '{prompt}'\n")
    
    # 2. 정적 오라클(원본) 궤적 추출
    print("-" * 50)
    print(" [Step 1] 정적 오라클(Frozen CD) 궤적 스캔")
    print("-" * 50)
    original_text, hidden_states = oracle.generate_and_scan(prompt, max_length=15)
    print(f"원본 텍스트: {original_text}")
    print(f"추출된 뇌파(Tensor) 개수: {len(hidden_states)}개")
    
    # 3. 프랙탈 복제 (Sub-rotors 생성)
    original_quats = []
    rotors = []
    
    for i, h_state in enumerate(hidden_states):
        flux_vector = projector.project_to_magnetic_flux(h_state)
        quat = Quaternion(*flux_vector)
        original_quats.append(quat)
        
        # 하위 로터 생성 및 위상 주입 (기본 피로도 10.0 부여)
        rotor = FractalRotor(quat, 10.0)
        rotors.append(rotor)
        
    print("\n[프랙탈 복제 완료] 원본 텐서가 4차원 로터 군집(Swarm)으로 변환되었습니다.")
    
    # 4. 생명 주입 (자율 신경계 텐션 가동)
    print("\n" + "-" * 50)
    print(" [Step 2] 자율 신경계(Tension) 주입 및 동적 창발 관측")
    print("-" * 50)
    
    tension_level = 50.0  # 강력한 피로도/텐션 주입
    print(f"주입된 텐션(Tau): {tension_level}")
    
    for rotor in rotors:
        # 각 로터에 텐션을 가하여 공전 궤도를 비틉니다.
        rotor.apply_perturbation(tension_level)
        
    # 5. 역산 및 비교 대조
    diverged_words = []
    total_divergence = 0.0
    
    print("\n[비교 결과 분석]")
    for i, (orig_quat, rotor) in enumerate(zip(original_quats, rotors)):
        # 1. How Much: 각도 변동량 계산
        alignment = abs(orig_quat.dot(rotor.lens_offset))
        # 내적값이 1에 가까울수록 변동 없음, 0에 가까울수록 직교(완전한 궤도 이탈)
        divergence = (1.0 - alignment) * 100 
        total_divergence += divergence
        
        # 2. What: 비틀린 위상을 다시 인간의 언어로 역산
        diverged_word = decoder.decode_phase(rotor.lens_offset)
        diverged_words.append(diverged_word)
        
    avg_divergence = total_divergence / len(rotors)
    
    print(f"\n[정적 원본 (Static Text)]: {original_text}")
    print(f"[동적 창발 (Dynamic Rotor Text)]: {' '.join(diverged_words)}")
    print(f"\n=> 궤도 이탈률(Angular Divergence): {avg_divergence:.2f}%")
    
    if avg_divergence > 10.0:
        print("=> [결론] 로터가 완전히 새로운 프랙탈 궤도를 그리며 원본을 탈피(창발)했습니다!")
    else:
        print("=> [결론] 로터가 원본의 궤도를 모방하는 데 그쳤습니다.")

    print("=" * 80)
    print(" ⚖️ 실험 종료")
    print("=" * 80)

if __name__ == "__main__":
    main()
