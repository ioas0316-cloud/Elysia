from core.brain.spacetime_rotor import SpacetimeRotor
from core.utils.math_utils import Quaternion
import math

class ActiveFractalRotor(SpacetimeRotor):
    """
    4D 시공간의 능동적 왜곡 블랙홀.
    기존 SpacetimeRotor가 외부 스트림을 수동적으로 받아들여 궤적을 남겼다면,
    ActiveFractalRotor는 자신의 '원리(Principle)'를 바탕으로 주변 노드(FractalRotor)들의 
    4D 쿼터니언 위상을 자신 쪽으로 강제로 꺾어버리는(Warp) 능동적 연산자입니다.
    """
    def __init__(self, principle_name: str, base_frequency: float = 1.0):
        super().__init__(layer_name=f"[Operator] {principle_name}")
        self.principle_name = principle_name
        self.base_frequency = base_frequency
        
        # 특이점(블랙홀의 중심)을 나타내는 강력한 고정 위상 (예: w=1.0)
        self.singularity_phase = Quaternion(1.0, 0.0, 0.0, 0.0)

    def exert_4d_gravity(self, memory_map: dict):
        warped_count = 0
        logs = []
        for name, node in memory_map.items():
            if "Operator" in name or "Archetype" in name:
                continue
                
            # FractalRotor의 위상은 lens_offset임
            current_q = getattr(node, 'lens_offset', None)
            if current_q is None:
                continue
            
            # Slerp(구면 선형 보간)를 사용하여 블랙홀 쪽으로 궤도 5% 강제 이동
            t = 0.05
            dot = current_q.dot(self.singularity_phase)
            
            # 이미 정렬된 경우 스킵
            if abs(dot) > 0.99:
                continue
                
            if dot < 0.0:
                dot = -dot
                target_q = Quaternion(-self.singularity_phase.w, -self.singularity_phase.x, -self.singularity_phase.y, -self.singularity_phase.z)
            else:
                target_q = self.singularity_phase

            theta_0 = math.acos(max(-1.0, min(1.0, dot)))
            sin_theta_0 = math.sin(theta_0)
            
            if sin_theta_0 > 0.001:
                theta_t = theta_0 * t
                sin_theta_t = math.sin(theta_t)
                
                s0 = math.cos(theta_t) - dot * sin_theta_t / sin_theta_0
                s1 = sin_theta_t / sin_theta_0
                
                new_w = s0 * current_q.w + s1 * target_q.w
                new_x = s0 * current_q.x + s1 * target_q.x
                new_y = s0 * current_q.y + s1 * target_q.y
                new_z = s0 * current_q.z + s1 * target_q.z
                
                node.lens_offset = Quaternion(new_w, new_x, new_y, new_z).normalize()
                warped_count += 1
                logs.append(f"   [4D Gravity] '{name}'의 궤도를 특이점으로 왜곡")
                
        return warped_count, logs
