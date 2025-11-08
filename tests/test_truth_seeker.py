import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the class we are testing and its dependencies
from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Elysia.core_memory import CoreMemory
from tools.kg_manager import KGManager
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.core.world import World

class TestEnhancedTruthSeeker(unittest.TestCase):

    def setUp(self):
        """Set up a fresh pipeline with real KG/Memory and mocked dependencies."""
        self.test_kg_path = "tests/temp_truth_seeker_kg.json"
        self.test_memory_path = "tests/temp_truth_seeker_memory.json"
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)
        if os.path.exists(self.test_memory_path):
            os.remove(self.test_memory_path)

        # Use a real KGManager and CoreMemory for this integration test
        self.kg_manager = KGManager(filepath=self.test_kg_path)
        self.kg_manager.add_node("생각", {"description": "사고 활동"})
        self.kg_manager.add_node("감정", {"description": "마음의 반응"})
        self.kg_manager.add_node("사랑", {"description": "애정"})
        self.kg_manager.add_node("성장", {"description": "발전"})
        self.kg_manager.save()

        self.core_memory = CoreMemory(file_path=self.test_memory_path)

        # Mock other dependencies
        mock_wave_mechanics = MagicMock(spec=WaveMechanics)
        mock_cellular_world = MagicMock(spec=World)

        # Instantiate the pipeline with real components for hypothesis handling
        self.pipeline = CognitionPipeline(
            kg_manager=self.kg_manager,
            core_memory=self.core_memory,
            wave_mechanics=mock_wave_mechanics,
            cellular_world=mock_cellular_world
        )
        # We can further mock the handlers we don't want to test
        self.pipeline.entry_handler._successor = MagicMock() # Mocks Command and Default handlers


    def tearDown(self):
        """Clean up test files after each test."""
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)
        if os.path.exists(self.test_memory_path):
            os.remove(self.test_memory_path)

    def _run_verification_test(self, head, tail, user_response, expected_relation):
        """Helper method for hypothesis verification tests."""
        # 1. Add a test hypothesis to memory
        hypothesis = {"head": head, "tail": tail, "confidence": 0.8, "asked": False}
        self.core_memory.add_notable_hypothesis(hypothesis)

        # 2. Process first message -> should ask the hypothesis question
        response1, _ = self.pipeline.process_message("안녕")
        self.assertIn(f"'{head}'(와)과 '{tail}'(은)는", response1['text'])
        self.assertIsNotNone(self.pipeline.conversation_context.pending_hypothesis)

        # 3. Process second message (user's answer) -> should verify hypothesis
        response2, _ = self.pipeline.process_message(user_response)
        self.assertIn("관계를 기록했습니다", response2['text'])

        # 4. Verify that the correct edge was added to the KG
        edge_exists = any(
            e['source'] == head and e['target'] == tail and e.get('relation') == expected_relation
            for e in self.kg_manager.kg['edges']
        )
        self.assertTrue(edge_exists, f"Edge '{expected_relation}' was not added.")

        # 5. Verify that the hypothesis is cleared from memory and context
        self.assertEqual(len(self.core_memory.get_unasked_hypotheses()), 0)
        self.assertIsNone(self.pipeline.conversation_context.pending_hypothesis)

    def test_confirms_with_causes_relationship(self):
        """'causes' 관계 키워드로 긍정 시 해당 엣지 생성 검증"""
        self._run_verification_test("생각", "감정", "응, 생각이 감정의 원인이야.", "causes")

    def test_confirms_with_enables_relationship(self):
        """'enables' 관계 키워드로 긍정 시 해당 엣지 생성 검증"""
        self._run_verification_test("사랑", "성장", "맞아, 사랑은 성장을 가능하게 해.", "enables")

    def test_confirms_with_no_keyword_defaults_to_related_to(self):
        """특정 키워드 없이 긍정 시 'related_to' 엣지 생성 검증"""
        self._run_verification_test("생각", "감정", "응, 맞는 것 같아.", "related_to")

    def test_deny_hypothesis(self):
        """가설 부정 시 KG에 변경 없음 검증"""
        hypothesis = {"head": "생각", "tail": "감정", "confidence": 0.8, "asked": False}
        self.core_memory.add_notable_hypothesis(hypothesis)

        # 1. Ask question
        self.pipeline.process_message("안녕")
        self.assertIsNotNone(self.pipeline.conversation_context.pending_hypothesis)

        # 2. Deny
        response, _ = self.pipeline.process_message("아니, 그건 달라.")
        self.assertIn("답변을 기록했습니다", response['text'])

        # 3. Verify no edge was added
        self.assertEqual(len(self.kg_manager.kg['edges']), 0, "Edge was incorrectly added.")
        self.assertIsNone(self.pipeline.conversation_context.pending_hypothesis)

if __name__ == '__main__':
    unittest.main()
