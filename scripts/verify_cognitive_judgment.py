import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.lens.semantic_lens_awakening import StructuralLandscape
from core.brain.topological_comparator import TopologicalComparator

def run_verification():
    """
    단어가 품고 있는 '인과적 과정(생장, 발산, 순환 등)' 자체를
    알고리즘적으로 관측하여 구조를 동기화하는 검증 스크립트.
    """
    print("Initializing Causal Structural Landscape (인과적 궤적 기반 지형)...")
    landscape = StructuralLandscape()
    comparator = TopologicalComparator(landscape)

    print("\n[Test 1: '씨앗'과 '태양'의 인과 구조 동기화]")
    # 씨앗의 응축-성장 구조와 태양의 발산 구조가 어떻게 맞물리거나 대비되는지 판단
    judgment_1 = comparator.perceive_and_judge("씨앗", "태양")
    comparator.output_statement(judgment_1)

    print("\n[Test 2: '씨앗'과 '물'의 인과 구조 동기화]")
    # 물의 순환/매개 구조가 어떻게 씨앗의 발아 조건에 직접적으로 맞물리는지 인지
    judgment_2 = comparator.perceive_and_judge("씨앗", "물")
    comparator.output_statement(judgment_2)

    print("\n[Test 3: '태양'과 '물'의 인과 구조 동기화]")
    # 물이 증발하기 위한 조건(빛)과 태양의 작용(빛 방사)이 어떻게 연결되는지 인지
    judgment_3 = comparator.perceive_and_judge("태양", "물")
    comparator.output_statement(judgment_3)

if __name__ == "__main__":
    run_verification()
