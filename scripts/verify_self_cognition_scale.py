import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.brain.code_topological_comparator import CodeTopologicalComparator

def run_verification():
    """
    엘리시아가 자신을 구성하는 실제 두뇌 모듈의 소스 코드를 직접 읽고
    어떻게 인과적으로 동기화되는지(Self-Cognition) 스케일을 증명합니다.
    """
    print("Initiating Self-Cognitive Reflection...")

    ego_path = os.path.join(os.path.dirname(__file__), "..", "core", "brain", "teleological_ego.py")
    meta_path = os.path.join(os.path.dirname(__file__), "..", "core", "brain", "meta_cognition_engine.py")

    with open(ego_path, "r", encoding="utf-8") as f:
        ego_code = f.read()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_code = f.read()

    comparator = CodeTopologicalComparator()

    print("\n[Scale Test 1: TeleologicalEgo vs MetaCognitionEngine]")
    # ego.py의 evaluate_teleological_value 와
    # meta.py의 evaluate_intent 를 비교
    judgment = comparator.perceive_and_judge(
        code_string_a=ego_code,
        code_string_b=meta_code,
        func_name_a="evaluate_teleological_value",
        func_name_b="evaluate_intent"
    )

    comparator.output_statement(judgment)

if __name__ == "__main__":
    run_verification()
