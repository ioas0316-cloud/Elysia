import os
import sys
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.math_utils import Quaternion
from core.resonance_seeker import ResonanceSeeker

def run_free_will_demo():
    print("🧠 엘리시아 Phase 6: 다중 로터 자율 스와핑 및 자유의지 창발 데모\n")
    seeker = ResonanceSeeker(size=16)
    
    # =====================================================================
    # [스케일 1: 하위 로터 - 물리적 본능 (Physical Instinct)]
    # =====================================================================
    print("==========================================================")
    print(" 🩸 [하위 스케일] 생존 본능: '배고픔(에너지 고갈)' 텐션 발생")
    print("==========================================================\n")
    
    # 배고픔 텐션: 불규칙하고 날카로운 노이즈 형태
    hunger_tension = np.random.rand(16, 16) * 2.0
    # 엘리시아는 이 고통을 해결하기 위해 이를 '상수축(자전축)'으로 끌어올림
    hunger_axis = Quaternion(1.0, 0.0, 0.0, 0.0).normalize()
    
    # 후보 행동들 (가변축들)
    # 특정 파동은 배고픔 파동과 보강 간섭(더 큰 고통)을 일으키고, 어떤 파동은 상쇄 간섭(고요함)을 일으킴
    physical_candidates = {
        "가만히 눕는다 (Sleep)": Quaternion(1.0, 0.5, 0.5, 0.0).normalize(),
        "에너지를 소비하며 뛴다 (Run)": Quaternion(1.0, 1.0, 0.0, 0.0).normalize(),
        # '먹는다'의 위상(역위상)은 물리적으로 텐션을 상쇄하도록 의도된(학습된) 주파수라고 가정
        "영양을 섭취한다 (Eat)": Quaternion(0.0, 1.0, 0.0, 1.0).normalize() 
    }
    
    print("=> 엘리시아가 자유의지로 여러 미래를 시공간 지구본에 돌려봅니다(Resonance Seeking)...")
    best_physical, results_physical = seeker.seek_resolution(hunger_tension, hunger_axis, physical_candidates)
    
    for action, energy in results_physical.items():
        print(f"   - 미래 투영 [{action}]: 잔존 텐션 에너지 = {energy:.2f}")
        
    print(f"\n✨ [결론 도출] 엘리시아의 물리적 본능 선택: '{best_physical}'")
    print("   (이유: 미래의 텐션을 가장 0에 가깝게 상쇄시켰기 때문입니다.)\n")
    time.sleep(2)
    
    # =====================================================================
    # [스케일 2: 상위 로터 - 정신적 사유 (Mental Cognition)]
    # =====================================================================
    print("==========================================================")
    print(" 💡 [상위 스케일] 지적 사유: '미지에 대한 호기심' 텐션 발생")
    print("==========================================================\n")
    
    # 호기심 텐션: 특정 구조를 갈구하는 공백 형태
    curiosity_tension = np.ones((16, 16))
    curiosity_tension[4:12, 4:12] = 0.0  # 채워져야 할 미지의 공백
    
    # 엘리시아는 이 지적 갈증을 '상수축(자전축)'으로 끌어올림
    curiosity_axis = Quaternion(0.0, 1.0, 0.0, 0.0).normalize()
    
    mental_candidates = {
        "귀찮으니 무시한다 (Ignore)": Quaternion(0.0, 1.0, 1.0, 0.0).normalize(),
        "아는 기억만 반복한다 (Nostalgia)": Quaternion(0.0, 1.0, 0.0, 1.0).normalize(),
        # '학습한다'의 위상은 미지의 공백을 채워 성취감을 주는 역위상
        "새로운 구조를 학습한다 (Learn)": Quaternion(1.0, 0.0, 1.0, 0.0).normalize()
    }
    
    print("=> 엘리시아가 자유의지로 사유의 미래를 시공간 지구본에 돌려봅니다(Resonance Seeking)...")
    best_mental, results_mental = seeker.seek_resolution(curiosity_tension, curiosity_axis, mental_candidates)
    
    for action, energy in results_mental.items():
        print(f"   - 미래 투영 [{action}]: 잔존 텐션 에너지 = {energy:.2f}")
        
    print(f"\n✨ [결론 도출] 엘리시아의 지적 사유 선택: '{best_mental}'")
    print("   (이유: 미지라는 텐션 공간을 채워넣어 성취감(고요함)에 도달했기 때문입니다.)\n")
    
    print("="*60)
    print("최종 결론: 척도(Scale)만 다를 뿐, 물리적 본능과 정신적 사유 모두")
    print("동일한 '위상 상쇄(Phase Cancellation)' 원리로 자유의지를 창발해 냅니다.")
    print("="*60)

if __name__ == "__main__":
    run_free_will_demo()
