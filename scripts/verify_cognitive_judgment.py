import os
import sys

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.lens.semantic_lens_awakening import SemanticLandscape
from core.brain.topological_comparator import TopologicalComparator

def run_verification():
    """
    엘리시아가 "언어를 기준 삼아 이해를 확장하는 사고방식"을
    실제로 수행할 수 있는지 검증하는 스크립트.
    """
    print("Initializing Semantic Landscape (의미망 기반 지형)...")
    landscape = SemanticLandscape()
    comparator = TopologicalComparator(landscape)

    print("\n[Test 1: 생명과 태양의 위상적 비교]")
    # '생명'과 '태양'은 직접적인 관계는 없어 보이지만, '빛'을 통해 인과적으로 연결될 수 있는지 확인
    judgment_1 = comparator.perceive_and_judge("생명", "태양")
    comparator.output_statement(judgment_1)

    print("\n[Test 2: 빛과 어둠의 위상적 비교]")
    # '어둠'의 정의 속에 '빛'이 결여되어 있다는 직접적 연결성이 잘 도출되는지 확인
    judgment_2 = comparator.perceive_and_judge("빛", "어둠")
    comparator.output_statement(judgment_2)

    print("\n[Test 3: 숲과 호흡의 위상적 비교]")
    # '숲'과 '호흡'이 어떻게 다르고, 어떻게 생명이라는 연결고리로 이어질 수 있는지 확인
    judgment_3 = comparator.perceive_and_judge("숲", "호흡")
    comparator.output_statement(judgment_3)


if __name__ == "__main__":
    run_verification()
