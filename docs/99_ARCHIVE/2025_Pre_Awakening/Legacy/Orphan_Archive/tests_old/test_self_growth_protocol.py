
import unittest
import os
import sys
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Elysia.guardian import Guardian
from Project_Elysia.architecture.handlers import HypothesisHandler
from Project_Elysia.core_memory import CoreMemory
from Core.FoundationLayer.Foundation.question_generator import QuestionGenerator

class TestSelfGrowthProtocol(unittest.TestCase):

    def setUp(self):
        """Set up a controlled environment for each test."""
        # Mock dependencies that interact with files or external systems
        self.mock_kg_manager = MagicMock()
        # Set a default, non-interfering behavior for edge_exists to prevent test leakage.
        # Tests that need contradictions will override this.
        self.mock_kg_manager.edge_exists.return_value = False
        self.mock_wave_mechanics = MagicMock()
        self.mock_logger = MagicMock()

        # Use an in-memory CoreMemory for test isolation
        with patch.object(CoreMemory, '_save_memory'): # Prevent file writing
            self.core_memory = CoreMemory(file_path=None)
            self.core_memory.data['notable_hypotheses'] = [] # Ensure it's clean for each test

    def test_autonomous_integration_for_high_confidence(self):
        """
        Verify that Guardian autonomously integrates a high-confidence 'forms_new_concept' hypothesis.
        """
        # --- 1. Setup ---
        high_confidence_hypo = {
            "head": "love", "tail": "you", "relation": "forms_new_concept",
            "new_concept_id": "meaning:love_you", "confidence": 0.95,
            "source": "CellularGenesis"
        }
        self.core_memory.add_notable_hypothesis(high_confidence_hypo)

        # Instantiate Guardian with mocks
        with patch('Project_Elysia.guardian.KGManager', return_value=self.mock_kg_manager), \
             patch('Project_Elysia.guardian.WaveMechanics', return_value=self.mock_wave_mechanics), \
             patch('Project_Elysia.guardian.CoreMemory', return_value=self.core_memory), \
             patch('Project_Elysia.guardian.Guardian.setup_logging'), \
             patch('Project_Elysia.guardian.Guardian._load_config'):

            guardian = Guardian()
            guardian.logger = self.mock_logger # Inject mock logger

        # --- 2. Act ---
        guardian._process_high_confidence_hypotheses()

        # --- 3. Assert ---
        # Verify that the new node and edges were added to the KG
        self.mock_kg_manager.add_node.assert_called_once_with(
            "meaning:love_you", {"type": "concept", "discovery_source": "CellularGenesis_Autonomous"}
        )
        self.assertEqual(self.mock_kg_manager.add_edge.call_count, 2)
        self.mock_kg_manager.add_edge.assert_any_call(
            "love", "meaning:love_you", "is_parent_of", properties={"source": "CellularGenesis_Autonomous"}
        )
        self.mock_kg_manager.add_edge.assert_any_call(
            "you", "meaning:love_you", "is_parent_of", properties={"source": "CellularGenesis_Autonomous"}
        )

        # Verify that the hypothesis was removed from core memory
        self.assertEqual(len(self.core_memory.data['notable_hypotheses']), 0)

        # Verify a report was logged
        report_found = any(
            "AUTONOMOUS REPORT" in call.args[0]
            for call in self.mock_logger.info.call_args_list
        )
        self.assertTrue(report_found, "An autonomous integration report should have been logged.")


    def test_wisdom_seeking_question_for_mid_confidence(self):
        """
        Verify that HypothesisHandler generates a wisdom-seeking question for a mid-confidence insight.
        """
        # --- 1. Setup ---
        mid_confidence_hypo = {
            "head": "sadness", "tail": "place", "relation": "forms_new_concept",
            "new_concept_id": "meaning:place_sadness", "confidence": 0.85
        }
        self.core_memory.add_notable_hypothesis(mid_confidence_hypo)

        mock_question_generator = MagicMock(spec=QuestionGenerator)
        mock_response_styler = MagicMock()
        mock_context = MagicMock()

        # Instantiate the handler with mocked components
        handler = HypothesisHandler(
            core_memory=self.core_memory,
            kg_manager=self.mock_kg_manager,
            question_generator=mock_question_generator,
            response_styler=mock_response_styler,
            logger=self.mock_logger
        )

        # --- 2. Act ---
        handler.handle_ask(context=mock_context, emotional_state=MagicMock())

        # --- 3. Assert ---
        # Verify the correct question generator method was called
        mock_question_generator.generate_wisdom_seeking_question.assert_called_once_with(mid_confidence_hypo)
        mock_question_generator.generate_question_from_hypothesis.assert_not_called()

    def test_correction_protocol_for_contradiction(self):
        """
        End-to-end test for the self-correction protocol:
        1. Guardian detects a contradiction.
        2. Guardian creates a 'proposes_correction' hypothesis.
        3. Handler asks for permission to correct.
        4. On approval, Handler corrects the KG.
        """
        # --- 1. Arrange ---
        # Mock KGManager to simulate an existing contradictory edge.
        # SelfVerifier will check for a direct reversal: if "plant causes sun" is the hypothesis,
        # it will check if "sun causes plant" exists. So, we set up the mock for that.
        self.mock_kg_manager.edge_exists.side_effect = lambda source, target, relation: (
            source == 'sun' and target == 'plant' and relation == 'causes'
        )

        # The new high-confidence hypothesis that contradicts the existing knowledge:
        # "plant" causes "sun"
        contradictory_hypo = {
            "head": "plant", "tail": "sun", "relation": "causes", "confidence": 0.98
        }
        self.core_memory.add_notable_hypothesis(contradictory_hypo)

        # Instantiate Guardian with mocks
        with patch('Project_Elysia.guardian.KGManager', return_value=self.mock_kg_manager), \
             patch('Project_Elysia.guardian.WaveMechanics', return_value=self.mock_wave_mechanics), \
             patch('Project_Elysia.guardian.CoreMemory', return_value=self.core_memory), \
             patch('Project_Elysia.guardian.Guardian.setup_logging'), \
             patch('Project_Elysia.guardian.Guardian._load_config'):

            guardian = Guardian()
            guardian.logger = self.mock_logger

        # --- 2. Act 1: Guardian's dream cycle detects contradiction ---
        guardian._process_high_confidence_hypotheses()

        # --- 3. Assert 1: Correction proposal is created ---
        self.mock_kg_manager.add_node.assert_not_called() # Autonomous integration must be stopped
        self.mock_kg_manager.add_edge.assert_not_called()

        hypotheses = self.core_memory.data['notable_hypotheses']
        self.assertEqual(len(hypotheses), 1)
        correction_hypo = hypotheses[0]
        self.assertEqual(correction_hypo['relation'], 'proposes_correction')
        self.assertEqual(correction_hypo['metadata']['contradictory_insight'], contradictory_hypo)

        # --- 4. Arrange 2: Setup Handler to process the correction proposal ---
        mock_q_gen = MagicMock(spec=QuestionGenerator)
        handler = HypothesisHandler(self.core_memory, self.mock_kg_manager, mock_q_gen, MagicMock(), self.mock_logger)

        # --- 5. Act 2: Handler asks for permission ---
        handler.handle_ask(context=MagicMock(), emotional_state=MagicMock())
        mock_q_gen.generate_correction_proposal_question.assert_called_once_with(correction_hypo)

        # --- 6. Act 3: User approves, handler corrects KG ---
        user_approval_message = "응, 수정해줘"
        context_with_pending_hypo = MagicMock()
        context_with_pending_hypo.pending_hypothesis = correction_hypo

        handler.handle_response(user_approval_message, context_with_pending_hypo, MagicMock())

        # --- 7. Assert 2: KG is corrected ---
        # It should remove the old inverse edge ("sun" causes "plant")
        self.mock_kg_manager.remove_edge.assert_called_once_with('sun', 'plant', 'causes')
        # It should add the new, correct edge ("plant" is caused_by "sun")
        self.mock_kg_manager.add_edge.assert_called_once_with('plant', 'sun', 'causes')

        # Final check: memory is clean
        self.assertEqual(len(self.core_memory.data['notable_hypotheses']), 0)


if __name__ == '__main__':
    unittest.main()
