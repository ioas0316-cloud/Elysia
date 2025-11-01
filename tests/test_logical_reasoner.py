import unittest
import os
import json
import sys
from pathlib import Path

# Add the project root to the Python path to resolve module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.cognition_pipeline import CognitionPipeline
from tools import kg_manager

class TestLogicalReasonerIntegration(unittest.TestCase):

    def setUp(self):
        """테스트 전에 환경을 설정합니다."""
        # 테스트 실행 전, 손상 가능성이 있는 메인 메모리 파일 삭제
        main_memory_path = 'Elysia_Input_Sanctum/elysia_core_memory.json'
        if os.path.exists(main_memory_path):
            os.remove(main_memory_path)

        # 테스트용 임시 KG 파일 경로 설정
        self.test_kg_path = Path('data/test_kg.json')
        # 모듈 레벨 변수에 접근
        self.original_kg_path = kg_manager.KG_PATH
        kg_manager.KG_PATH = self.test_kg_path

        # 임시 지식 그래프에 테스트 데이터 추가
        # 클래스를 모듈을 통해 접근
        self.kg_manager_instance = kg_manager.KGManager()
        self.kg_manager_instance.add_node("소크라테스")
        self.kg_manager_instance.add_node("인간")
        self.kg_manager_instance.add_edge("소크라테스", "인간", "is_a")
        self.kg_manager_instance.save()

        # CognitionPipeline 인스턴스 생성
        self.pipeline = CognitionPipeline()

    def tearDown(self):
        """테스트 후에 환경을 정리합니다."""
        # 원래 KG 경로로 복원
        kg_manager.KG_PATH = self.original_kg_path
        # 임시 KG 파일 삭제
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)

    def test_reasoning_and_response(self):
        """논리 추론과 그에 따른 응답 생성이 올바르게 작동하는지 테스트합니다."""
        # 테스트할 메시지
        test_message = "소크라테스에 대해 알려줘"

        # 파이프라인을 통해 메시지 처리
        response, _ = self.pipeline.process_message(test_message)

        # More robust check
        self.assertIn("소크라테스", response)
        self.assertIn("인간", response)
        self.assertIn("is_a", response)
        print("\n--- 테스트 통과 ---")
        print(f"입력: '{test_message}'")
        print(f"응답: {response}")
        print("-------------------")

if __name__ == '__main__':
    unittest.main()
