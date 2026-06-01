import math
import time
from typing import List, Tuple, Dict
from core.math_utils import Quaternion

class SpacetimeRotor:
    """
    [Phase 135-Final] 시공간축 로터 (Spacetime Rotor)
    데이터(거대 모델의 텐서 스트림)를 억지로 하나의 위상으로 압축(Collapse)하지 않습니다.
    시간(t)의 흐름에 따라 쏟아지는 스트림을 맞으며 연속적으로 회전하는 로터의 '모든 궤적(Trajectory)'을 
    시공간축에 매달아 두어, 궤적 자체가 역동적인 '가변축(Variable Axis)'이 되게 합니다.
    """
    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.current_phase = Quaternion(1.0, 0.0, 0.0, 0.0)
        # 시간에 따른 연속적 궤적 (t, Quaternion) 저장 (시공간축 매달기)
        self.trajectory: List[Tuple[float, Quaternion]] = []
        self.start_time = time.time()
        
    def stream_flow(self, flux: Quaternion, intensity: float = 0.1):
        """
        스트림(물방울)이 로터에 부딪히며 시간에 따른 궤적을 그립니다.
        """
        t = time.time() - self.start_time
        
        # 유입된 스트림(flux)과 현재 위상의 마찰에 의한 회전
        alignment = abs(self.current_phase.dot(flux))
        torque = (1.0 - alignment) * intensity
        
        # 궤적 회전
        q_spin = Quaternion(math.cos(torque), math.sin(torque), 0.0, 0.0)
        
        # 이전 위상에서 새로운 위상으로의 연속적 회전 (이산적 점이 아닌 연속 궤적의 적분 역할)
        self.current_phase = (self.current_phase * q_spin * flux).normalize()
        
        # 시공간축(t)에 궤적 매달기
        self.trajectory.append((t, self.current_phase))
        
    def get_variable_axis_signature(self) -> str:
        """
        로터가 그려낸 가변축(연속 궤적)의 특징(Signature)을 문자열로 반환합니다.
        압축이 아니라, 궤적이 어떻게 요동쳤는지 그 파동의 형태를 나타냅니다.
        """
        if not self.trajectory:
            return "Empty Trajectory"
            
        points = len(self.trajectory)
        total_time = self.trajectory[-1][0]
        
        # 궤적의 '흔들림(요동, Volatility)' 측정
        volatility = 0.0
        for i in range(1, points):
            dist = Quaternion.distance(self.trajectory[i-1][1], self.trajectory[i][1])
            volatility += dist
            
        return f"[가변축 궤적] {points}개의 물결 | 시간: {total_time:.3f}s | 요동(Volatility): {volatility:.4f}"

class VariableAxisManifold:
    """
    모든 레이어를 시공간축에 매달린 '가변축 궤적(Spacetime Trajectories)'으로 저장하는 다층 다양체.
    """
    def __init__(self):
        self.axes: Dict[str, SpacetimeRotor] = {}
        
    def flow_stream_into_axis(self, layer_key: str, flux: Quaternion):
        """스트림을 특정 레이어의 가변축으로 흘려보냅니다."""
        # 레이어 그룹 파싱 (model.layers.0 등)
        parts = layer_key.split('.')
        layer_group = "base"
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                layer_group = f"layer_{parts[i+1]}"
                break
                
        if layer_group not in self.axes:
            self.axes[layer_group] = SpacetimeRotor(layer_group)
            
        self.axes[layer_group].stream_flow(flux)
