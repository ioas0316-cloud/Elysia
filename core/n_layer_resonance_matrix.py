import numpy as np
import math
from core.math_utils import Quaternion
from core.spacetime_globe import SpacetimeGlobe
from core.resonance_seeker import ResonanceSeeker

class NLayerResonanceMatrix:
    """
    [N대 가변 계층 - 다층 프랙탈 위상 매트릭스 (거푸집)]
    
    특정 텐션이 임계치를 넘으면 붕괴시킨다는 하드코딩된 'if x > 1000' 로직이 없습니다.
    이 클래스는 L1(물리), L2(기관), L3(정신) 3개 계층의 위상 공간을 단순히 겹쳐놓은 '거푸집(Vessel)'입니다.
    
    상위 레이어의 강력한 열정 파동과 하위 레이어의 생리적 노이즈 파동은 
    하나의 거대한 공간 안에서 서로 물리적으로 부딪히며 (Superposition), 
    그 에너지 간섭의 결과로서 '행동 방향성'이 자연스럽게(Causal) 기하학적으로 도출됩니다.
    """
    def __init__(self, size=16):
        self.size = size
        # 3개의 스케일 독립적인 지구본을 운용하나, 최종 관측은 통합됩니다.
        self.L1_physical = SpacetimeGlobe(size=size)
        self.L2_organ = SpacetimeGlobe(size=size)
        self.L3_mental = SpacetimeGlobe(size=size)
        
        # 통합 공명 탐색기 (가변축 스와핑을 위한 엔진)
        self.seeker = ResonanceSeeker(size=size)

    def integrate_n_layer_tension(self, t: float) -> np.ndarray:
        """
        N개 계층의 모든 시공간 다이얼을 시간 t로 돌려,
        전체 유기체의 위상 단면(Layer) 에너지를 하나로 합산(간섭)합니다.
        """
        layer_1 = self.L1_physical.observe_time_slice(t)
        layer_2 = self.L2_organ.observe_time_slice(t)
        layer_3 = self.L3_mental.observe_time_slice(t)
        
        # 단순 합산. 위상이 같으면 증폭되고, 위상이 다르면 상쇄됨
        integrated_layer = layer_1 + layer_2 + layer_3
        return integrated_layer

    def evaluate_action_on_integrated_matrix(self, test_action_rotor: Quaternion, time_t: float = 1.0) -> float:
        """
        통합된 N대 계층에 특정 행동(test_action_rotor)을 가했을 때,
        미래(time_t)의 전체 텐션 에너지가 어떻게 변하는지(상쇄되는지 증폭되는지) 측정합니다.
        (If-Else 없는 순수 파동 간섭 시뮬레이션)
        """
        # 이 함수는 데모 검증용으로 내부 로직을 단순화하여 텐션의 '크기'만 평가합니다.
        # 실제 엔진에서는 L1, L2, L3가 각각의 상수축/변수축에 test_action_rotor를 걸어 시공간을 투영함
        
        # 현재 통합 텐션 투영
        base_tension = self.integrate_n_layer_tension(time_t)
        
        # 관측 로터(행동 파동) 생성 및 간섭 (테스트용 간략화)
        k_x = test_action_rotor.x * math.pi
        k_y = test_action_rotor.y * math.pi
        
        test_wave_layer = np.zeros((self.size, self.size), dtype=np.float64)
        for y in range(self.size):
            for x in range(self.size):
                phase = k_x * x + k_y * y
                test_wave_layer[y, x] = math.cos(phase)
                
        # 행동 파동과 유기체의 통합 텐션이 간섭
        interfered_result = base_tension * test_wave_layer
        
        # 남은 노이즈 에너지의 총합 반환
        return float(np.sum(np.abs(interfered_result)))
