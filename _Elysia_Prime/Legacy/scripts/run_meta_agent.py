# [Genesis: 2025-12-02] Purified by Elysia
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Project_Elysia.high_engine.meta_agent import MetaAgent


def run_meta_agent() -> None:
    print("\n--- [엘리시아 메타 에이전트: 자율 확장 루프] ---\n")
    agent = MetaAgent()
    actions = agent.cycle()

    if actions.get("meta_law"):
        print("[완료] 메타 법칙 축을 재정의하고 KG에 연동했습니다.")
    else:
        print("[유지] 새로운 메모리 개념이 발견되지 않았습니다.")

    if actions.get("grammar"):
        print("[완료] 문법 학습을 다시 실행했습니다.")
    else:
        print("[유지] 문법 모델은 최신 상태입니다.")


if __name__ == "__main__":
    run_meta_agent()