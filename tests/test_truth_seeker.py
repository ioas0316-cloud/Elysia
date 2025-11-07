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

class TestEnhancedTruthSeeker(unittest.TestCase):

    def setUp(self):
        """테스트 시작 전에 매번 호출되는 메소드"""
        self.test_kg_path = "tests/temp_truth_seeker_kg.json"
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)

        self.kg_manager = KGManager(filepath=self.test_kg_path)
        self.kg_manager.add_node("생각", {"description": "사고 활동"})
        self.kg_manager.add_node("감정", {"description": "마음의 반응"})
        self.kg_manager.add_node("사랑", {"description": "애정"})
        self.kg_manager.add_node("성장", {"description": "발전"})

        self.core_memory = CoreMemory(file_path="tests/temp_truth_seeker_memory.json")

        # CognitionPipeline 인스턴스화 및 의존성 주입
        self.pipeline = CognitionPipeline()
        self.pipeline.core_memory = self.core_memory
        self.pipeline.kg_manager = self.kg_manager

        patcher = patch('random.random', return_value=0.1)
        self.mock_random = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        """테스트 종료 후 임시 파일 정리"""
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)
        if os.path.exists("tests/temp_truth_seeker_memory.json"):
            os.remove("tests/temp_truth_seeker_memory.json")

    def _run_verification_test(self, head, tail, user_response, expected_relation):
        """가설 검증 테스트를 위한 헬퍼 메소드"""
        # 1. 테스트용 가설 추가
        hypothesis = {"head": head, "tail": tail, "confidence": 0.8, "asked": False}
        self.core_memory.add_notable_hypothesis(hypothesis)

        # 2. 가설 질문 단계 실행
        self.pipeline.process_message("안녕")
        self.assertIsNotNone(self.pipeline.pending_hypothesis_verification)

        # 3. 사용자 답변 처리
        self.pipeline.process_message(user_response)

        # 4. KG에 특정 관계 엣지가 추가되었는지 확인
        edge_exists = any(
            e['source'] == head and e['target'] == tail and e['relation'] == expected_relation
            for e in self.kg_manager.kg['edges']
        )
        self.assertTrue(edge_exists, f"'{expected_relation}' 엣지가 추가되지 않았습니다.")

        # 5. 처리된 가설이 메모리에서 제거되었는지 확인
        self.assertEqual(len(self.core_memory.data['notable_hypotheses']), 0)
        self.assertIsNone(self.pipeline.pending_hypothesis_verification)

    def test_confirms_with_causes_relationship(self):
        """'causes' 관계 키워드로 긍정 시 해당 엣지 생성 검증"""
        self._run_verification_test("생각", "감정", "응, 생각이 감정의 원인이야.", "causes")

    def test_confirms_with_enables_relationship(self):
        """'enables' 관계 키워드로 긍정 시 해당 엣지 생성 검증"""
        self._run_verification_test("사랑", "성장", "맞아, 사랑은 성장을 가능하게 해.", "enables")

    def test_confirms_with_supports_relationship(self):
        """'supports' 관계 키워드로 긍정 시 해당 엣지 생성 검증"""
        self._run_verification_test("사랑", "성장", "응 사랑이 성장에 도움이 돼.", "supports")

    def test_confirms_with_no_keyword_defaults_to_related_to(self):
        """특정 키워드 없이 긍정 시 'related_to' 엣지 생성 검증"""
        self._run_verification_test("생각", "감정", "응, 맞는 것 같아.", "related_to")

    def test_deny_hypothesis(self):
        """가설 부정 시 KG에 변경 없음 검증"""
        hypothesis = {"head": "생각", "tail": "감정", "confidence": 0.8, "asked": False}
        self.core_memory.add_notable_hypothesis(hypothesis)

        self.pipeline.process_message("안녕")
        self.assertIsNotNone(self.pipeline.pending_hypothesis_verification)

        self.pipeline.process_message("아니, 그건 달라.")

        # KG에 어떠한 엣지도 추가되지 않았는지 확인
        self.assertEqual(len(self.kg_manager.kg['edges']), 0, "엣지가 잘못 추가되었습니다.")
        self.assertEqual(len(self.core_memory.data['notable_hypotheses']), 0)
        self.assertIsNone(self.pipeline.pending_hypothesis_verification)

if __name__ == '__main__':
    unittest.main()
