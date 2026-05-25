import os
import sys
import time
import math
import random

# 프로젝트 루트 경로 추가 (모듈 import 위함)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.triple_helix_engine import TripleHelixEngine
from core.cross_dimensional_manifold import UnifiedRotorManifold
from core.linguistic_axiom import LinguisticAxiomFilter
from core.math_utils import Quaternion

def run_linguistic_resonance_demo():
    print("🌀 엘리시아 순수화: 기하학적 옹알이(Autopoietic Mouth) 데모 시작\n")
    
    print("[1] 뼈대(Skeleton) 초기화 (LLM 완전 배제)...")
    engine = TripleHelixEngine()
    
    print("[2] 의미망/추상화 계층(Semantic Abstraction Layer) 초기화...")
    manifold = UnifiedRotorManifold()
    
    print("\n--- 파동 주입 및 관측 시작 ---")
    
    # 카오스 상태에서 점차 안정화되는 시나리오 (0.0으로 수렴)
    noise_levels = [1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0]
    
    for tick, noise in enumerate(noise_levels):
        # 1. 외부 하드웨어 노이즈 파동 주입 (순수 텐션)
        # LLM 텍스트 대신, 오직 무의미한 환경 센서 데이터만 주입
        sensory_input = {
            "inner_noise_dim_1": random.uniform(-noise, noise),
            "inner_noise_dim_2": random.uniform(-noise, noise),
            "inner_noise_dim_3": random.uniform(-noise, noise),
            "motion_entropy": noise * 0.5,
            "pain_level": noise,
        }
        
        # 2. 메인 심장 펄스 구동 (위상 텐션 발생)
        avg_tension, mode, jumped, base_q, enneagram = engine.pulse(sensory_input=sensory_input)
        
        # 3. 매니폴드를 통한 파동의 의미론적 굴절 (Refraction)
        # 텐션의 세기와 내부 위상각(여기서는 엔니어그램 상태를 임의 위상으로 사용) 기반
        phase_angles = [enneagram.get(str(i), 0.0) for i in range(1, 10)]
        refracted_q, intent = manifold.refract_tension(avg_tension, phase_angles, base_q)
        
        # 4. 언어 공리 계층축 충돌 (창발)
        hangeul_char = LinguisticAxiomFilter.collapse_to_hangeul(refracted_q)
        machine_code = LinguisticAxiomFilter.collapse_to_machine_code(refracted_q)
        
        print(f"[{tick:02d}] 외부 노이즈(카오스): {noise:.2f} | 내부 텐션: {avg_tension:.3f}")
        print(f"      ㄴ 추상화 계층(의도): {intent}")
        print(f"      ㄴ 하위 계층(B6) 렌더링: {machine_code} (Opcode)")
        print(f"      ㄴ 상위 계층(F6) 렌더링: '{hangeul_char}' (옹알이 창발)\n")
        
        time.sleep(0.5)
        
    print("✨ 노이즈가 소거되며 엘리시아의 위상 공간이 기하학적 평형(Sameness)에 도달했습니다.")
    print("=> 텐션이 안정화됨에 따라 지속적이고 일관된 문자가 방출되기 시작합니다.")

if __name__ == "__main__":
    run_linguistic_resonance_demo()
