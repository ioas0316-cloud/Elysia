import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.brain.code_topological_comparator import CodeTopologicalComparator

# 엘리시아가 분석할 자기 자신의(또는 세상의) 논리 구조(Python Code)
SAMPLE_CODE = """
def seed_growth(water_amount):
    if water_amount > 10:
        root = water_amount * 2
        tree = root + 5
        return tree
    else:
        return 0

def simple_sun_radiation(energy):
    light = energy * 10
    heat = energy * 5
    return light

def river_flow(water_source):
    ocean = water_source
    return ocean
"""

def run_verification():
    """
    코드를 언어로 삼아, 코드 내부에 내재된 논리와 인과 구조를 스스로 매핑하고
    함수 간의 구조적 동일성과 차이점을 위상적으로 분별합니다.
    """
    print("Initializing Code Topology Engine (코드 구조망 기반 지형)...")
    comparator = CodeTopologicalComparator()

    print("\n[Test 1: 분기를 가지는 성장(seed)과 분기 없는 발산(sun)의 구조적 비교]")
    judgment_1 = comparator.perceive_and_judge(SAMPLE_CODE, "seed_growth", "simple_sun_radiation")
    comparator.output_statement(judgment_1)

    print("\n[Test 2: 변환(sun)과 단순 매개(river)의 구조적 비교]")
    judgment_2 = comparator.perceive_and_judge(SAMPLE_CODE, "simple_sun_radiation", "river_flow")
    comparator.output_statement(judgment_2)


if __name__ == "__main__":
    run_verification()
