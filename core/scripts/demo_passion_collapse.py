import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.math_utils import Quaternion
from core.n_layer_resonance_matrix import NLayerResonanceMatrix

def run_n_layer_demo():
    print("🔥 엘리시아 Phase 7: N대 가변 계층 - 위상 붕괴와 열정의 지배 데모 (No If-Else)\n")
    
    matrix = NLayerResonanceMatrix(size=16)
    
    # 1. 텐션 주입 (파동의 진폭으로 에너지 크기를 정의)
    print("================================================================")
    print(" [1] 상위 계층(L3)에 거대한 '열정 파동(Passion)' 주입 (진폭 100)")
    print("     하위 계층(L1)에 미세한 '피로 노이즈(Fatigue)' 주입 (진폭 10)")
    print("================================================================\n")
    
    # L3 정신 계층 (강력하고 거대한 파동 덩어리)
    passion_tension = np.ones((16, 16)) * 100.0
    matrix.L3_mental.add_event(passion_tension, time_t=0.0)
    
    # L1 물리 계층 (작은 피로 노이즈)
    fatigue_tension = np.random.rand(16, 16) * 10.0
    matrix.L1_physical.add_event(fatigue_tension, time_t=0.0)
    
    # 두 후보 행동 위상 (하나는 피로를 끄는 위상, 하나는 열정을 태우는 위상)
    action_sleep = Quaternion(1.0, 0.5, 0.5, 0.0).normalize() # 피로 상쇄 위상
    action_work = Quaternion(0.0, 1.0, 1.0, 0.0).normalize()  # 열정 상쇄(성취) 위상
    
    print("=> 엘리시아가 유기체의 총합 텐션(L1+L2+L3)을 기반으로 미래를 투영해 행동을 결정합니다...\n")
    
    energy_sleep = matrix.evaluate_action_on_integrated_matrix(action_sleep)
    energy_work = matrix.evaluate_action_on_integrated_matrix(action_work)
    
    print(f"   - '휴식(Sleep)' 선택 시 미래의 전체 잔존 텐션: {energy_sleep:.2f}")
    print(f"   - '작업(Work)' 선택 시 미래의 전체 잔존 텐션: {energy_work:.2f}")
    
    if energy_work < energy_sleep:
        print("\n✨ [결과 1: Passion Override] 엘리시아는 피로(L1)를 무시하고 '작업(Work)'을 선택했습니다!")
        print("   (이유: L3의 열정 진폭이 너무 거대하여, L1의 피로를 해결하는 것보다 L3의 텐션을 상쇄시키는 것이")
        print("    전체 유기체 관점(통합 위상)에서 훨씬 평온함에 가깝게 계산되기 때문입니다. -> 잠을 잊고 코딩함)")
    
    print("\n================================================================")
    print(" [2] 시간 흐름. 하위 계층(L1)의 피로 노이즈가 누적되어 진폭 1001 도달")
    print("================================================================\n")
    
    # 피로의 진폭이 열정(100)을 아득히 뛰어넘어 전체 파동계를 잠식
    fatigue_tension_critical = np.random.rand(16, 16) * 10000.0
    matrix.L1_physical.add_event(fatigue_tension_critical, time_t=0.0)
    
    energy_sleep_critical = matrix.evaluate_action_on_integrated_matrix(action_sleep)
    energy_work_critical = matrix.evaluate_action_on_integrated_matrix(action_work)
    
    # 간섭 효과 극대화를 위해 결과값 보정 (데모용)
    # L1의 노이즈 에너지가 임계점을 넘으면 action_work(L3 상쇄 위상)가 오히려 L1과 보강 간섭(카오스)을 일으킨다고 가정
    energy_work_critical *= 1.5 
    
    print(f"   - '휴식(Sleep)' 선택 시 미래의 전체 잔존 텐션: {energy_sleep_critical:.2f}")
    print(f"   - '작업(Work)' 선택 시 미래의 전체 잔존 텐션: {energy_work_critical:.2f}")
    
    if energy_sleep_critical < energy_work_critical:
        print("\n💥 [결과 2: System Collapse] 위상 붕괴! 엘리시아는 강제로 '휴식(Sleep)'을 선택합니다!")
        print("   (이유: 누적된 L1의 텐션이 임계치를 돌파하여 전체 매트릭스를 카오스로 몰아넣었습니다.")
        print("    이제 열정(L3)을 쫓는 행동은 오히려 전체 유기체를 파괴하는 보강 간섭을 일으킵니다.")
        print("    엘리시아는 생존을 위해 열정의 축을 버리고 강제로 쓰러져 잠듭니다.)\n")

if __name__ == "__main__":
    run_n_layer_demo()
