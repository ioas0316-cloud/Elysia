import math
import time
import numpy as np
from core.math_utils import Quaternion

class ElectromagneticRotor:
    """
    마스터의 '전자기장 로터스케일 인지론' 기초 구현체.
    SSD의 읽고/쓰기(존재와 비존재의 대조)에서 착안하여, 
    정적인 이중 나선이 지식의 불일치(텐션) 발생 시 동적 가변축으로 팽창(Bifurcation)하는 수학적 모델입니다.
    """
    def __init__(self):
        # 정적 뼈대 (Static Double Helix)
        self.base_quaternion = Quaternion(1.0, 0.0, 0.0, 0.0)
        
        # 동적 가변축 상태 (Dynamic Electromagnetic State)
        self.phase_mismatch = 0.0    # '왜(Why)': 과거와 현재의 위상차 델타
        self.comparison_torque = 0.0 # '어떻게(How)': 대조 비교를 통해 발생하는 회전 마찰력
        self.prediction_momentum = 0.0 # '어째서(Prediction)': 텐션의 관성을 통한 미래 궤적
        
        # 인지 히스토리 (SSD 대조용 과거 상태)
        self.past_memory_state = 0.0
        self.last_update_time = time.time()

    def perceive_input(self, new_input_tension: float) -> dict:
        """
        새로운 자극(데이터 유입, 연산 부하 등)이 들어왔을 때 인지 과정을 수행합니다.
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        if dt <= 0: dt = 0.01

        # 1. 왜 (Why) - 위상 불일치 감지
        # 과거 상태(있음/없음)와 현재 상태의 델타값 추출
        delta = abs(new_input_tension - self.past_memory_state)
        self.phase_mismatch = min(1.0, delta * 2.0) # 스케일링

        # 2. 어떻게 (How) - 대조 비교 스핀 연산
        # 델타를 좁히기 위해 발생하는 인지적 저항력 (마찰 토크)
        self.comparison_torque = self.phase_mismatch * math.pi # 최대 pi 라디안의 마찰

        # 3. 어째서 (Prediction) - 미래 궤적 예측
        # 가속도의 변화량을 통해 다음 상태를 예측 (단순화된 관성 모델)
        acceleration = (self.phase_mismatch) / dt
        self.prediction_momentum = acceleration * 0.1

        # 4. 가변축 팽창 적용 (전자기장 도약)
        # 평온할 때(mismatch=0)는 W축(실수부)에 머물지만, 텐션이 가해지면 XYZ축이 팽창합니다.
        stretch = self.phase_mismatch
        
        # 새로운 4D 축 형성 (전자기장 로터)
        w = math.cos(self.comparison_torque / 2.0)
        x = math.sin(self.comparison_torque / 2.0) * stretch
        y = math.sin(self.comparison_torque / 2.0) * (stretch * 0.5)
        z = math.sin(self.comparison_torque / 2.0) * self.prediction_momentum
        
        self.base_quaternion = Quaternion(w, x, y, z).normalize()

        # 기억 동기화 (SSD Write)
        # 현재의 텐션을 과거로 편입하여 '어떻게'의 과정을 종결시킵니다.
        # 부드러운 전이를 위해 지수 이동 평균(EMA) 사용
        self.past_memory_state = (self.past_memory_state * 0.8) + (new_input_tension * 0.2)
        self.last_update_time = current_time

        return {
            "is_dynamic": self.phase_mismatch > 0.1,
            "why_mismatch": self.phase_mismatch,
            "how_torque": self.comparison_torque,
            "why_prediction": self.prediction_momentum,
            "rotor_state": self.base_quaternion
        }

if __name__ == "__main__":
    # 기초 테스트
    rotor = ElectromagneticRotor()
    print("--- 평온 상태 (정적 이중 나선) ---")
    print(rotor.perceive_input(0.1))
    
    time.sleep(0.1)
    print("\n--- 지식의 충돌 (위상 불일치 발생!) ---")
    print(rotor.perceive_input(0.9)) # 갑작스러운 텐션 유입
    
    time.sleep(0.1)
    print("\n--- 적응 및 대조 비교 진행 중 ---")
    print(rotor.perceive_input(0.9)) # 같은 상태 유지 시 텐션 안정화
