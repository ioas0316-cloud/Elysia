import sys
import os

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.math_utils import Multivector
from core.clifford_impedance_network import mv_normalize, mv_norm
import math

def test_pure_bivector_alignment():
    print("=======================================")
    print(" 🌀 Rotor Backpropagation / Outermorphism")
    print("=======================================")
    
    # 1. 초기 상태 S와 목표 상태 T 설정 (Cl(3,0) 공간)
    # S는 무작위로 치우쳐 있고, T는 완전 다른 방향
    S = mv_normalize(Multivector({1: 1.0, 2: 0.2, 4: -0.5}, (3, 0))) # e1 중심
    T = mv_normalize(Multivector({1: -0.2, 2: 1.0, 4: 0.8}, (3, 0))) # e2, e3 중심
    
    # 기어의 초기 상태 (Identity 모터)
    M = Multivector({0: 1.0}, (3, 0))
    
    elasticity = 0.2 # 스칼라 학습률(lr)을 대체하는 물리적 기어 탄성계수
    
    coherence_history = []
    
    print(f"초기 S: {S.data}")
    print(f"초기 T: {T.data}")
    
    for step in range(20):
        # 1. 모터 샌드위치 연산을 통해 신호 전파: S_out = M S ~M
        # M은 정규화된 로터이므로 크기 보존, 방향 회전 수행
        S_out = mv_normalize(M * S * M.conjugate())
        
        # 2. 전파된 신호와 타겟 간의 기하곱 병렬 동기화 추출
        coherence, B = T.geometric_sync(S_out)
        coherence_history.append(coherence)
        
        # 3. [미적분 박멸] 스칼라 경사하강법 없이, 오차 평면 B를 향해 모터 자체를 물리적으로 회전시킴
        # 쐐기곱 토크 B를 모터의 오프셋으로 주입 (Rotor Spin)
        # exp(B * elasticity) ~ 1 + B * elasticity
        M_step = Multivector({0: 1.0}, (3, 0)) + B * elasticity
        M = mv_normalize(M_step * M)
        
        if step % 5 == 0:
            print(f"[Step {step:02d}] Coherence (Alignment): {coherence:.4f}, Torque |B|: {mv_norm(B):.4f}")
            
    print(f"[Final Step 20] Coherence (Alignment): {coherence_history[-1]:.4f}")
    
    assert coherence_history[-1] > 0.99, "기하곱 회전 역전파만으로 위상 동기화(Phase Lock)에 실패했습니다!"
    print("\n✅ 수학 벤치마크 PASS: 오직 Bivector Torque(기어 장력)만으로 타겟 궤도에 100% 안착했습니다. 미적분 박멸 완료!")

if __name__ == "__main__":
    test_pure_bivector_alignment()
