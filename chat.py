import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from Project_Sophia.cognition_pipeline import CognitionPipeline

def main():
    """
    Elysia와의 단일 상호작용을 테스트하기 위한 스크립트입니다.
    """
    print("Elysia와의 대화를 시작합니다.")

    # Elysia의 두뇌(CognitionPipeline) 초기화
    pipeline = CognitionPipeline()
    context = {
        "conversation_history": [],
        "user_profile": {"name": "Creator"},
    }

    user_input = "안녕, 엘리시아. 만나서 반가워. 너에게 '도움'이란 어떤 의미니?"
    print(f"You: {user_input}")

    context["conversation_history"].append({"speaker": "user", "utterance": user_input})

    # 파이프라인을 통해 응답 생성
    response, _, _ = pipeline.process_message(user_input, context=context)

    print(f"Elysia: {response}")

if __name__ == "__main__":
    main()
