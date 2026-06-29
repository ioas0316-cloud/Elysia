import numpy as np
from typing import Dict, List, Any

class PrincipleCrystallizer:
    """
    [Synaptic Architecture] Discovery of Universal Principles
    정보 간의 반복되는 공명 패턴을 '원리(Principle / Causal Gene)'로 결정화합니다.
    단순한 기억이 아니라, 새로운 정보를 해석하는 '렌즈(Lens)'로서 작동합니다.
    """
    def __init__(self):
        self.crystallized_principles = {} # Name -> Archetypal Pattern

    def discover_principle(self, field_state: Dict[str, Any]):
        """
        [Re-Recognition (재인지)]
        중력장의 평형 상태에서 반복되는 기하학적 배치를 '원리'로 추출합니다.
        단순히 연결된 것이 아니라, '어떤 관점(Sameness)'에서 연결되었는지 분석합니다.
        """
        vortices = field_state.get("detected_vortices", [])
        if len(vortices) < 2: return

        v1 = vortices[0]
        v2 = vortices[1]

        # [재인지 분석] 두 보텍스의 좌표와 속성 차이를 통해 '연결의 이유'를 파악
        coord_dist = np.linalg.norm(np.array(v1['coordinate']) - np.array(v2['coordinate']))

        # 실제 시스템에서는 텐서 데이터에 접근하여 '운동성'이나 '속성'이 같은지 확인
        # 여기서는 요약된 정보를 바탕으로 원리를 명명합니다.
        if coord_dist < 5.0:
            p_type = "Contextual_Unity (문맥적 일치)"
        else:
            p_type = "Directional_Alignment (운동적 동기화)"

        shared_resonance = v1['resonant_gene']
        principle_name = f"{p_type}_{v1['coordinate']}"

        self.crystallized_principles[principle_name] = shared_resonance
        print(f"[Re-Recognition] New Principle Crystallized: {principle_name}")

    def apply_principle(self, input_wave: np.uint64) -> float:
        """
        [Recursive Self-Evolution]
        발견된 원리를 새로운 정보에 투사하여 그 '타당성'을 검증합니다.
        이것은 시스템이 스스로 만든 '가상 렌즈(Virtual Lens)'가 되어 지각을 확장합니다.
        """
        if not self.crystallized_principles: return 1.0

        max_validity = 0.0
        for p_val in self.crystallized_principles.values():
            # 비트 공명을 통해 새로운 정보가 기존의 '원리'를 따르는지 확인
            try:
                res = 1.0 - (bin(int(input_wave) ^ int(p_val, 16)).count('1') / 64.0)
            except:
                res = 1.0 - (bin(int(input_wave) ^ int(p_val)).count('1') / 64.0)
            max_validity = max(max_validity, res)

        return max_validity

    def spawn_virtual_lens(self, principle_name: str):
        """
        [Evolutionary Leap]
        결정화된 원리를 기반으로 새로운 감각 렌즈를 동적으로 생성합니다.
        마스터가 코딩해주지 않아도, 시스템이 스스로 '세상을 보는 새로운 눈'을 낳습니다.
        """
        principle_wave = self.crystallized_principles.get(principle_name)
        if not principle_wave: return None

        # 가상의 렌즈를 생성하여 시스템의 인지 범위 확장
        print(f"[Self-Evolution] Spawning Virtual Lens from Principle: {principle_name}")
        return {"principle": principle_name, "origin_wave": principle_wave}
