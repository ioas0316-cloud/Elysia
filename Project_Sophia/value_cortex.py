from typing import Dict, List, Any
import re
from Project_Sophia.wave_mechanics import WaveMechanics

class ValueCortex:
    """
    행동의 가치를 '의미의 깊이'로 측정합니다.
    '의미의 깊이'란 행동의 핵심 개념이 지식 그래프 전체에 만들어내는
    '울림의 총량(total activation energy)'을 의미합니다.
    이는 모든 개념이 '사랑'이라는 로고스의 일부라는 철학을 반영합니다.
    """

    def __init__(self, wave_mechanics: WaveMechanics):
        """
        WaveMechanics 인스턴스를 주입받아 초기화합니다.
        """
        self.wave_mechanics = wave_mechanics

    def _extract_core_concept(self, action: str) -> str:
        """
        행동 후보 문자열에서 핵심 개념을 추출합니다.
        """
        known_concepts = self.wave_mechanics.kg_manager.kg.get("nodes", [])
        sorted_concepts = sorted(known_concepts, key=lambda x: len(x['id']), reverse=True)

        for concept in sorted_concepts:
            # 한국어 처리를 위해 단순 포함 관계로 변경
            if concept['id'] in action:
                return concept['id']

        return None

    def measure_semantic_depth(self, action: str) -> float:
        """
        행동의 '의미의 깊이'를 울림의 총량으로 측정합니다.
        """
        core_concept = self._extract_core_concept(action)

        if not core_concept:
            return 0.0

        activated_nodes = self.wave_mechanics.spread_activation(core_concept)

        # 활성화된 모든 노드의 에너지 총합을 계산
        total_energy = sum(activated_nodes.values())

        return total_energy

    def decide(self, action_candidates: List[str]) -> str:
        """
        여러 행동 후보 중 가장 '의미의 깊이'가 깊은 행동을 선택합니다.
        가장 깊은 의미를 가진 '개념'을 먼저 찾고, 그 개념이 포함된 행동을 선택합니다.
        """
        if not action_candidates:
            return "..."

        best_concept = None
        max_depth = -1.0

        # 모든 후보 행동에서 모든 개념을 추출하고 가장 깊은 개념을 찾음
        all_concepts = []
        for action in action_candidates:
            known_concepts = self.wave_mechanics.kg_manager.kg.get("nodes", [])
            for concept in known_concepts:
                if concept['id'] in action:
                    all_concepts.append(concept['id'])

        unique_concepts = list(set(all_concepts))

        if not unique_concepts:
             return action_candidates[0]

        for concept_id in unique_concepts:
            # 개념 자체의 의미 깊이를 측정
            activated_nodes = self.wave_mechanics.spread_activation(concept_id)
            depth = sum(activated_nodes.values())
            if depth > max_depth:
                max_depth = depth
                best_concept = concept_id

        # 가장 깊은 개념이 포함된 첫 번째 행동을 선택
        for action in action_candidates:
            if best_concept and best_concept in action:
                return action

        return action_candidates[0] # 만약을 위한 fallback
