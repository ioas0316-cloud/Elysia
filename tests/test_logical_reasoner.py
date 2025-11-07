import unittest
import os
import json
import sys
from pathlib import Path

# Add the project root to the Python path to resolve module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Elysia.cognition_pipeline import CognitionPipeline
from tools.kg_manager import KGManager

class TestLogicalReasonerIntegration(unittest.TestCase):

    def setUp(self):
        """테스트 전에 환경을 설정합니다."""
        # 테스트 실행 전, 손상 가능성이 있는 메인 메모리 파일 삭제
        main_memory_path = 'Elysia_Input_Sanctum/elysia_core_memory.json'
        if os.path.exists(main_memory_path):
            os.remove(main_memory_path)

        # 테스트용 임시 KG 파일 경로 설정
        self.test_kg_path = Path('data/test_logical_reasoner_kg.json')
        if self.test_kg_path.exists():
            self.test_kg_path.unlink()

        # Create a KGManager instance specifically for this test
        self.kg_manager_instance = KGManager(filepath=self.test_kg_path)

        self.kg_manager_instance.add_node("소크라테스")
        self.kg_manager_instance.add_node("인간")
        self.kg_manager_instance.add_edge("소크라테스", "인간", "is_a")
        self.kg_manager_instance.save()

        # Since CognitionPipeline creates its own KGManager, we must patch it
        # to use our test-specific one.
        self.pipeline = CognitionPipeline()
        self.pipeline.kg_manager = self.kg_manager_instance

    def tearDown(self):
        """테스트 후에 환경을 정리합니다."""
        if self.test_kg_path.exists():
            self.test_kg_path.unlink()

    def test_reasoning_and_response(self):
        """논리 추론과 그에 따른 응답 생성이 올바르게 작동하는지 테스트합니다."""
        # 테스트할 메시지
        test_message = "소크라테스에 대해 알려줘"

        # 파이프라인을 통해 메시지 처리
        response, _ = self.pipeline.process_message(test_message)

        # More robust check for natural language response
        self.assertIn("소크라테스", response['text'])
        self.assertIn("인간", response['text'])
        self.assertIn("의 한 종류예요", response['text'])
        print("\n--- 테스트 통과 ---")
        print(f"입력: '{test_message}'")
        print(f"응답: {response}")
        print("-------------------")

if __name__ == '__main__':
    unittest.main()
