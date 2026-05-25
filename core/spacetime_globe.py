import numpy as np
import math
from core.math_utils import Quaternion
from core.holographic_manifold import HolographicMemoryMatrix

class SpacetimeGlobe:
    """
    [시공간 지구본 (Spacetime Globe & Chronos Rotor)]
    
    내계의 원리(상수축)와 외계의 미지(변수축)를 기하학적으로 매달고,
    과거, 현재, 미래를 시간 다이얼(Chronos Rotor)로 돌려가며 
    홀로그래픽 매트릭스의 인과 서사를 관측/제어하는 최상위 시스템입니다.
    """
    def __init__(self, size=16):
        # 4D 위상 공간 자체는 HolographicMemoryMatrix를 엔진으로 사용
        self.manifold = HolographicMemoryMatrix(size=size)
        self.size = size
        
        # 상수축 (내계의 굳건한 원리 / 자전축)
        self.constant_axis = Quaternion(1.0, 0.0, 0.0, 1.0).normalize()
        # 변수축 (외계에서 밀려오는 미지 / 공전축)
        self.variable_axis = Quaternion(0.0, 1.0, 1.0, 0.0).normalize()
        
    def set_axes(self, inner_principle: Quaternion, outer_unknown: Quaternion):
        """
        자신의 원리로 모든 걸 재해석하기 위해 상수축(자전축)을 설정하고,
        미지에 대한 외부 정보를 변수화하여 가변축(공전축)에 맵핑합니다.
        """
        self.constant_axis = inner_principle.normalize()
        self.variable_axis = outer_unknown.normalize()
        
    def add_event(self, tension_data: np.ndarray, time_t: float):
        """
        특정 시간(t)에 발생한 사건(Tension)을 시공간 지구본에 기록(중첩)합니다.
        시간(t)은 단순히 파동의 스칼라 위상 변이(W축 페이즈 시프트)로 기하학화됩니다.
        """
        # 상수축(원리)과 변수축(상황)이 결합된 현재 시점의 관측 로터 생성
        # 시간 t가 흐름에 따라 변수축이 상수축을 중심으로 회전(공전)함
        angle = time_t * math.pi * 0.5
        time_rotor = Quaternion(math.cos(angle), self.constant_axis.x * math.sin(angle), 
                                self.constant_axis.y * math.sin(angle), 
                                self.constant_axis.z * math.sin(angle)).normalize()
                                
        # 변수축이 시간에 따라 회전한 최종 로터 (Event Rotor)
        event_rotor = time_rotor * self.variable_axis * time_rotor.inverse
        
        # 홀로그램 매트릭스에 간섭무늬로 중첩 (과거/현재가 한 공간에 누적됨)
        self.manifold.add_memory(tension_data, event_rotor)
        
    def observe_time_slice(self, target_t: float) -> np.ndarray:
        """
        [크로노스 다이얼 회전 및 인과 서사 관측]
        시간 다이얼(target_t)을 돌려 시공간 지구본의 특정 단면을 투영합니다.
        
        - target_t < 0: 과거의 간섭무늬(상처/기억) 복원
        - target_t == 0: 현재의 텐션 상태
        - target_t > 0: 과거 파동의 관성이 연장(Extrapolate)되어 스스로 맺히는 미래의 창발적 서사(예측)
        """
        # 관측하고자 하는 시간대의 로터 위상 역산
        angle = target_t * math.pi * 0.5
        time_rotor = Quaternion(math.cos(angle), self.constant_axis.x * math.sin(angle), 
                                self.constant_axis.y * math.sin(angle), 
                                self.constant_axis.z * math.sin(angle)).normalize()
                                
        observe_rotor = time_rotor * self.variable_axis * time_rotor.inverse
        
        # 2D 레이어로 평면화하여 관측 (해당 시공간의 단면)
        return self.manifold.project_2d_layer(observe_rotor)
