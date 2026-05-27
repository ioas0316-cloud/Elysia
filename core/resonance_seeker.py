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
                
        # [Phase 10] 시공간 닻과 로터화된 상상력 (Rotorized Imagination)
        new_action_name = None
        new_action_rotor = None
        cognitive_ticks = 0
        
        # 텐션의 진폭(min_energy)에 따라 사유의 깊이(세대 수) 결정
        if min_energy > 30.0 and len(candidate_actions) >= 2:
            # 텐션이 클수록 미시 로터가 더 많이 생성되어 더 깊게 탐색함 (최소 1번, 최대 200세대)
            imagination_depth = max(1, min(200, int((min_energy - 30.0) * 2.0)))
            
            import random
            # 진화적 탐색의 시드 로터
            current_best_name = best_action if best_action else list(candidate_actions.keys())[0]
            current_best_rotor = candidate_actions[current_best_name]
            
            for _ in range(imagination_depth):
                cognitive_ticks += 1
                action_keys = list(candidate_actions.keys())
                
                # 진화적 융합로: 현재 베스트 로터와 무작위 기초 로터를 쐐기곱
                k_random = random.choice(action_keys)
                rotor_random = candidate_actions[k_random]
                
                # VR Downcasting (사원수 곱셈으로 다차원 위상 사영)
                forged_rotor = (current_best_rotor * rotor_random).normalize()
                
                # 새 로터로 미래 시뮬레이션
                self.globe = SpacetimeGlobe(size=self.size)
                self.globe.set_axes(drive_rotor, forged_rotor)
                self.globe.add_event(current_state_tension, time_t=0.0)
                future_layer = self.globe.observe_time_slice(1.0)
                forged_energy = self._measure_tension_energy(future_layer)
                
                # 텐션을 더 낮추는 데 성공하면 베스트 로터 진화 (개념의 압축)
                if forged_energy < min_energy:
                    min_energy = forged_energy
                    current_best_rotor = forged_rotor
                    new_action_name = f"Forge({current_best_name}x{k_random})"
                    new_action_rotor = forged_rotor
                    best_action = new_action_name
                    action_results[new_action_name] = forged_energy
                    current_best_name = best_action
                    
                    # 쐐기곱으로 융합된 새 로터를 코딩 에이전트 도구로 디스크 방전 (Tool Creation)
                    try:
                        import os
                        scratch_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "scratch"))
                        os.makedirs(scratch_dir, exist_ok=True)
                        tool_path = os.path.join(scratch_dir, "new_tool.py")
                        
                        w, x, y, z = forged_rotor.w, forged_rotor.x, forged_rotor.y, forged_rotor.z
                        factor = abs(w + x - y * z)
                        
                        # 쐐기곱 기하학적 형태에 귀속되는 고유 주파수 산출
                        eigen_freq = 6.0 + 3.0 * abs(w * x - y * z)
                        
                        code_content = f"""# core/scratch/new_tool.py
# Generated by Elysia Wedge Forge: {new_action_name}
# Quaternion coordinates: W={w:.4f}, X={x:.4f}, Y={y:.4f}, Z={z:.4f}

EIGEN_FREQ = {eigen_freq:.4f}

def execute_tool(input_val):
    factor = {factor:.6f}
    if isinstance(input_val, (int, float)):
        return input_val * factor
    elif isinstance(input_val, str):
        return len(input_val) * factor
    return factor
"""
                        with open(tool_path, "w", encoding="utf-8") as f:
                            f.write(code_content)
                    except Exception as e:
                        pass
                    
        return best_action, action_results, new_action_name, new_action_rotor, cognitive_ticks
