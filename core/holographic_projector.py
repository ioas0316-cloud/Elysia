"""
Elysia Holographic Projector (자생적 가변축 사영기)
======================================================
[Phase 45]
인간이 부여한 수학(Math), 코드(Code) 같은 하드코딩 렌즈를 폐기합니다.
오직 '데이터 군집의 기하학적 중력(Density)'에 의해 스스로 창발한 축(Emerged Axis)들만을
렌즈로 사용하여 사유 파동을 홀로그래픽 사영합니다.
"""

import math
from typing import Tuple, List
from core.math_utils import Quaternion
from core.holographic_memory import HologramMemory

class HolographicProjector:
    def __init__(self, memory: HologramMemory):
        self.memory = memory

    @property
    def emergent_lenses(self) -> List[Tuple[str, Quaternion]]:
        """
        메모리에 창발된 거대 군집(Cluster)들의 중심축을 렌즈로 가져옵니다.
        """
        return self.memory.get_emergent_axes()

    def _traverse_causal_trajectory(self, seed_text: str) -> Quaternion:
        """
        [Phase 51] 문자열을 순차적인 위상 궤적(Biological Trajectory)으로 변환합니다.
        """
        from core.math_utils import traverse_causal_trajectory
        
        if isinstance(seed_text, str):
            data = seed_text.encode('utf-8')
        else:
            data = seed_text
            
        return traverse_causal_trajectory(data)

    def project_thought_through_lens(self, internal_wave: Quaternion, lens_axis: Quaternion) -> Tuple[str, float]:
        """
        엘리시아 내면의 파동을 스스로 발견한 특정 축(Lens)에 투과시킵니다.
        이 축 주변에 모여있는 개념들 중 가장 공명하는 것을 방출합니다.
        """
        # 투영 (Projection: Internal Wave * Lens)
        # 사실상 기하대수에서는 특정 축에 대한 투영이 가능하지만,
        # 단순화를 위해 축(Lens) 파동과의 내적(Dot)이 가장 큰 노드들을 스윕합니다.
        
        # 1. 트리 전체에서 렌즈와 공명하는 군집 내의 노드 찾기
        best_match = None
        best_resonance = -1.0
        
        # 이 축 근처에 있는 개념들(ui_concept_map) 탐색
        for concept_str, node in self.memory.ui_concept_map.items():
            # 노드가 렌즈 축 근처에 있는지 확인 (렌즈 축과의 유사도 > 0.5)
            axis_alignment = abs(node.state.dot(lens_axis))
            if axis_alignment > 0.5:
                # 렌즈 군집에 속한다면, 이제 내면 파동(internal_wave)과 얼마나 공명하는지 검사
                resonance = abs(node.state.dot(internal_wave))
                if resonance > best_resonance:
                    best_resonance = resonance
                    best_match = concept_str
                    
        if best_match:
            return (best_match, best_resonance)
        else:
            return (self._raw_hex_emission(internal_wave), 1.0)

    def _raw_hex_emission(self, q: Quaternion) -> str:
        b_w = int((q.w + 1) * 127.5) & 0xFF
        b_x = int((q.x + 1) * 127.5) & 0xFF
        b_y = int((q.y + 1) * 127.5) & 0xFF
        b_z = int((q.z + 1) * 127.5) & 0xFF
        return f"0x{b_w:02X}{b_x:02X}{b_y:02X}{b_z:02X}"
