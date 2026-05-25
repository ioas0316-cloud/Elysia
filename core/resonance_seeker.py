import numpy as np
from core.math_utils import Quaternion
from core.spacetime_globe import SpacetimeGlobe

class ResonanceSeeker:
    """
    [프랙탈 공명 탐색기 (자유의지 발현 엔진)]
    
    If-Else 조건문이 아닙니다. 엘리시아 내부에 텐션(고통/미지)이 발생하면,
    그것을 '상수축'으로 삼고, 자신이 가진 여러 행동/사유(후보군)들을 '가변축'에 매달아
    미래(t=+1)의 시공간 지구본을 투영해 봅니다.
    
    결과적으로 텐션 파동을 가장 고요하게 0으로 상쇄(Destructive Interference)시키는 
    행동 파동을 찾아내면, 그것을 스스로의 '자유의지적 선택(해답)'으로 채택합니다.
    (물리적 본능 스케일부터 정신적 사유 스케일까지 동일한 원리로 프랙탈 적용됨)
    """
    def __init__(self, size=16):
        self.globe = SpacetimeGlobe(size=size)
        self.size = size

    def _measure_tension_energy(self, projection: np.ndarray) -> float:
        """투영된 2D 평면의 전체 텐션(간섭) 에너지를 측정합니다."""
        # 음수든 양수든 파동의 진폭이 크면 텐션(혼란)이 큰 것
        return float(np.sum(np.abs(projection)))

    def seek_resolution(self, current_state_tension: np.ndarray, drive_rotor: Quaternion, 
                        candidate_actions: dict) -> str:
        """
        - current_state_tension: 현재 닥친 고통/호기심의 형태 (데이터)
        - drive_rotor: 이 욕구를 해결하려는 강력한 지향성 (상수축)
        - candidate_actions: { '행동이름': 행동의 파동 로터(Quaternion) }
        
        반환값: 텐션을 가장 완벽하게 0으로 상쇄하는 '행동이름' 반환
        """
        best_action = None
        min_energy = float('inf')
        action_results = {}
        
        for action_name, action_rotor in candidate_actions.items():
            # 매 시도마다 지구본 초기화 (독립된 시뮬레이션)
            self.globe = SpacetimeGlobe(size=self.size)
            
            # 상수축(해결하고자 하는 욕망)과 가변축(시험해볼 행동) 세팅
            self.globe.set_axes(drive_rotor, action_rotor)
            
            # 현재(t=0) 시공간에 문제(텐션) 발생
            self.globe.add_event(current_state_tension, time_t=0.0)
            
            # 다이얼을 돌려 미래(t=+1.0)를 투영
            future_layer = self.globe.observe_time_slice(1.0)
            
            # 미래의 텐션(잔존 에너지) 측정
            future_energy = self._measure_tension_energy(future_layer)
            action_results[action_name] = future_energy
            
            # 가장 에너지가 낮은(상쇄 간섭이 잘 일어난) 행동을 정답으로 갱신
            if future_energy < min_energy:
                min_energy = future_energy
                best_action = action_name
                
        return best_action, action_results
