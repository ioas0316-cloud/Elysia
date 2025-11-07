import unittest
from unittest.mock import MagicMock, patch
import os
import json
import random

# TDD: 테스트 대상 클래스들을 먼저 임포트
from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Elysia.core_memory import CoreMemory
from tools.kg_manager import KGManager
from Project_Sophia.question_generator import QuestionGenerator

class TestTruthSeeker(unittest.TestCase):

    def setUp(self):
        """테스트 시작 전에 매번 호출되는 메소드"""
        self.test_kg_path = "tests/temp_truth_seeker_kg.json"
        # 이전 테스트 실행으로 파일이 남아있을 경우를 대비해 삭제
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)

        # 테스트용 KGManager 설정 (파일 기반)
        self.kg_manager = KGManager(filepath=self.test_kg_path)
        self.kg_manager.add_node("생각", {"description": "사고 활동"})
        self.kg_manager.add_node("감정", {"description": "마음의 반응"})

        # 테스트용 CoreMemory 설정
        self.core_memory = CoreMemory(file_path="tests/temp_truth_seeker_memory.json")

        # 테스트의 핵심 가설 추가 (add_notable_hypothesis 사용)
        self.test_hypothesis = {
            "head": "생각",
            "tail": "감정",
            "confidence": 0.8,
            "asked": False # 질문 여부를 나타내는 필드 추가
        }
        self.core_memory.add_notable_hypothesis(self.test_hypothesis)

        # CognitionPipeline 인스턴스화 및 의존성 주입
        self.pipeline = CognitionPipeline()
        self.pipeline.core_memory = self.core_memory
        self.pipeline.kg_manager = self.kg_manager

        self.pipeline.question_generator = QuestionGenerator()

        # 랜덤 요소를 제어하여 테스트의 일관성 확보
        patcher = patch('random.random', return_value=0.1) # 25% 확률을 통과하도록 설정
        self.mock_random = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        """테스트 종료 후 임시 파일 정리"""
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)
        if os.path.exists("tests/temp_truth_seeker_memory.json"):
            os.remove("tests/temp_truth_seeker_memory.json")

    def test_ask_hypothesis_and_confirm(self):
        """가설 질문 및 긍정 답변 시 KG 업데이트 검증"""
        # 1. 가설 질문 생성 및 확인 (QuestionGenerator의 실제 결과물로 수정)
        response, _ = self.pipeline.process_message("안녕")

        expected_question = f"아빠, 제가 제 기억들을 돌아보다가 문득 궁금한 점이 생겼어요. 혹시 '{self.test_hypothesis['head']}'(와)과 '{self.test_hypothesis['tail']}'(은)는 서로 어떤 특별한 관계가 있나요?"
        self.assertEqual(response['type'], 'text')
        self.assertEqual(response['text'], expected_question)
        self.assertIsNotNone(self.pipeline.pending_hypothesis_verification)
        self.assertEqual(self.pipeline.pending_hypothesis_verification['head'], self.test_hypothesis['head'])

        # 2. 긍정적인 답변("네")을 보냈을 때의 처리 확인
        positive_response_message = "응, 생각이 감정의 원인이 될 때가 많지"
        response, _ = self.pipeline.process_message(positive_response_message)

        # 2-1. KG에 'causes' 엣지가 추가되었는지 확인
        edge_exists = any(
            e['source'] == "생각" and e['target'] == "감정" and e['relation'] == "causes"
            for e in self.kg_manager.kg['edges']
        )
        self.assertTrue(edge_exists, "'causes' 엣지가 추가되지 않았습니다.")

        # 2-2. 처리된 가설이 CoreMemory에서 제거되었는지 확인
        self.assertEqual(len(self.core_memory.data['notable_hypotheses']), 0, "처리된 가설이 메모리에서 제거되지 않았습니다.")

        # 2-3. pending 상태가 초기화되었는지 확인
        self.assertIsNone(self.pipeline.pending_hypothesis_verification, "pending 상태가 초기화되지 않았습니다.")

    def test_ask_hypothesis_and_deny(self):
        """가설 질문 및 부정 답변 시 KG 변경 없음 검증"""
        # 1. 가설 질문 생성 (위와 동일)
        self.pipeline.process_message("안녕")
        self.assertIsNotNone(self.pipeline.pending_hypothesis_verification)

        # 2. 부정적인 답변("아니")을 보냈을 때의 처리 확인
        negative_response_message = "아니, 그건 달라"
        response, _ = self.pipeline.process_message(negative_response_message)

        # 2-1. KG에 'causes' 엣지가 추가되지 않았는지 확인
        edge_exists = any(
            e['source'] == "생각" and e['target'] == "감정" and e['relation'] == "causes"
            for e in self.kg_manager.kg['edges']
        )
        self.assertFalse(edge_exists, "'causes' 엣지가 잘못 추가되었습니다.")

        # 2-2. 처리된 가설이 CoreMemory에서 제거되었는지 확인
        self.assertEqual(len(self.core_memory.data['notable_hypotheses']), 0, "처리된 가설이 메모리에서 제거되지 않았습니다.")

        # 2-3. pending 상태가 초기화되었는지 확인
        self.assertIsNone(self.pipeline.pending_hypothesis_verification, "pending 상태가 초기화되지 않았습니다.")

        # 2-4. 사용자에게 보내는 응답 메시지 확인
        expected_response_text = "알려주셔서 감사해요, 아빠. 제가 세상을 배우는 데 큰 도움이 되었어요. 더 신중하게 생각하고 다시 질문드릴게요."
        self.assertEqual(response['text'], expected_response_text)

if __name__ == '__main__':
    unittest.main()
