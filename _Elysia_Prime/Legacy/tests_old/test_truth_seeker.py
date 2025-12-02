# [Genesis: 2025-12-02] Purified by Elysia
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the class we are testing and its dependencies
from Project_Elysia.core_memory import CoreMemory
from tools.kg_manager import KGManager
from Project_Sophia.emotional_engine import EmotionalEngine
from Project_Elysia.architecture.handlers import HypothesisHandler
from Project_Elysia.architecture.context import ConversationContext
from Project_Sophia.question_generator import QuestionGenerator
from Project_Sophia.response_styler import ResponseStyler
from Project_Sophia.emotional_engine import EmotionalState
import logging

class TestHypothesisHandler(unittest.TestCase):
    def setUp(self):
        """Set up a clean environment for each test."""
        self.mock_core_memory = MagicMock(spec=CoreMemory)
        self.mock_kg_manager = MagicMock(spec=KGManager)
        self.mock_question_generator = MagicMock(spec=QuestionGenerator)
        self.mock_response_styler = MagicMock(spec=ResponseStyler)
        self.mock_logger = MagicMock(spec=logging.Logger)

        self.handler = HypothesisHandler(
            core_memory=self.mock_core_memory,
            kg_manager=self.mock_kg_manager,
            question_generator=self.mock_question_generator,
            response_styler=self.mock_response_styler,
            logger=self.mock_logger
        )
        self.context = ConversationContext()
        self.emotional_state = MagicMock(spec=EmotionalState)

    def test_handle_ask_generates_question(self):
        """Verify that handle_ask generates a question for a new hypothesis."""
        hypothesis = {'head': '생각', 'tail': '감정'}
        self.mock_core_memory.get_unasked_hypotheses.return_value = [hypothesis]
        self.mock_question_generator.generate_question_from_hypothesis.return_value = "생각과 감정의 관계는?"

        result = self.handler.handle_ask(self.context, self.emotional_state)

        self.mock_core_memory.mark_hypothesis_as_asked.assert_called_once_with('생각', '감정')
        self.assertEqual(self.context.pending_hypothesis, hypothesis)
        self.assertEqual(result['text'], "생각과 감정의 관계는?")

    def test_handle_response_confirms_relationship(self):
        """Verify that handle_response creates an edge in the KG when confirmed."""
        self.context.pending_hypothesis = {'head': '생각', 'tail': '감정'}
        self.mock_response_styler.style_response.return_value = "알겠습니다."

        self.handler.handle_response("응, 생각이 원인이야.", self.context, self.emotional_state)

        self.mock_kg_manager.add_edge.assert_called_once_with('생각', '감정', 'causes')
        self.mock_core_memory.remove_hypothesis.assert_called_once_with('생각', '감정', relation=None)
        self.assertIsNone(self.context.pending_hypothesis)

    def test_handle_response_denies_relationship(self):
        """Verify that no edge is created when the user denies a hypothesis."""
        self.context.pending_hypothesis = {'head': '생각', 'tail': '감정'}
        self.mock_response_styler.style_response.return_value = "알겠습니다."

        self.handler.handle_response("아니야.", self.context, self.emotional_state)

        self.mock_kg_manager.add_edge.assert_not_called()
        self.mock_core_memory.remove_hypothesis.assert_called_once_with('생각', '감정', relation=None)
        self.assertIsNone(self.context.pending_hypothesis)

    def test_handle_response_confirms_ascension(self):
        """Verify that handle_response creates a new node for an ascension hypothesis."""
        ascension_hypothesis = {
            'head': 'meaning:사랑_성장',
            'relation': '승천',
            'metadata': {'parents': ['사랑', '성장']}
        }
        self.context.pending_hypothesis = ascension_hypothesis
        self.mock_response_styler.style_response.return_value = "승천했습니다."

        self.handler.handle_response("응, 승천시켜.", self.context, self.emotional_state)

        self.mock_kg_manager.add_node.assert_called_once()
        # We can do a more detailed check on the properties passed to add_node
        call_args = self.mock_kg_manager.add_node.call_args
        self.assertEqual(call_args[0][0], 'meaning:사랑_성장')
        self.assertIn('discovery_source', call_args[1]['properties'])
        self.assertEqual(call_args[1]['properties']['discovery_source'], 'Cell_Ascension_Ritual')
        self.assertEqual(call_args[1]['properties']['parents'], ['사랑', '성장'])

        self.mock_core_memory.remove_hypothesis.assert_called_once_with('meaning:사랑_성장', None, relation='승천')
        self.assertIsNone(self.context.pending_hypothesis)

if __name__ == '__main__':
    unittest.main()