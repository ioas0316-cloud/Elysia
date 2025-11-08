import unittest
from unittest.mock import MagicMock, patch, ANY
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.arithmetic_cortex import ArithmeticCortex
from Project_Elysia.cognition_pipeline import CognitionPipeline

# Import classes to be mocked as dependencies
from tools.kg_manager import KGManager
from Project_Elysia.core_memory import CoreMemory
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.core.world import World


class TestArithmeticCortex(unittest.TestCase):

    def setUp(self):
        self.cortex = ArithmeticCortex()
        # Pipeline is no longer initialized here to avoid dependency issues.
        # It will be created specifically in the integration tests that need it.

    def test_safe_addition(self):
        """Test basic addition."""
        response = self.cortex.process("5 + 3")
        self.assertEqual(response, "계산 결과는 8 입니다.")

    def test_safe_subtraction(self):
        """Test basic subtraction."""
        response = self.cortex.process("10 - 4")
        self.assertEqual(response, "계산 결과는 6 입니다.")

    def test_safe_multiplication(self):
        """Test basic multiplication."""
        response = self.cortex.process("6 * 7")
        self.assertEqual(response, "계산 결과는 42 입니다.")

    def test_safe_division(self):
        """Test basic division."""
        response = self.cortex.process("20 / 4")
        self.assertEqual(response, "계산 결과는 5.0 입니다.")

    def test_division_by_zero(self):
        """Test division by zero error handling."""
        response = self.cortex.process("10 / 0")
        self.assertEqual(response, "0으로 나눌 수 없습니다.")

    def test_invalid_expression(self):
        """Test invalid expression error handling."""
        response = self.cortex.process("5 +")
        self.assertEqual(response, "계산 형식을 이해하지 못했습니다. '계산: [수식]' 형태로 요청해주세요.")

    def test_unsafe_expression_import(self):
        """Test detection of unsafe code (import)."""
        response = self.cortex.process("__import__('os').system('echo unsafe')")
        self.assertIn("안전하지 않은 문자가 포함되어 계산할 수 없습니다", response)

    def test_unsafe_expression_variable(self):
        """Test detection of unsafe code (variables)."""
        response = self.cortex.process("a + b")
        self.assertIn("안전하지 않은 문자가 포함되어 계산할 수 없습니다", response)

    def test_korean_command(self):
        """Test handling of Korean commands."""
        response = self.cortex.process("계산: 100 + 200")
        self.assertEqual(response, "계산 결과는 300 입니다.")

    def test_pipeline_integration_question(self):
        """Test the pipeline integration for calculation questions."""
        # Setup mock dependencies for pipeline
        mock_kg_manager = MagicMock(spec=KGManager)
        mock_core_memory = MagicMock(spec=CoreMemory)
        mock_wave_mechanics = MagicMock(spec=WaveMechanics)
        mock_cellular_world = MagicMock(spec=World)
        mock_core_memory.get_unasked_hypotheses.return_value = [] # No pending hypotheses

        pipeline = CognitionPipeline(
            kg_manager=mock_kg_manager,
            core_memory=mock_core_memory,
            wave_mechanics=mock_wave_mechanics,
            cellular_world=mock_cellular_world
        )
        # Mock the successor of the command handler to isolate its behavior
        pipeline.entry_handler._successor._successor = MagicMock()

        response, _ = pipeline.process_message("계산: 10 + 5")
        self.assertEqual(response['text'], "계산 결과는 15 입니다.")

    def test_pipeline_integration_command(self):
        """Test the pipeline integration for calculation commands."""
        # Setup mock dependencies for pipeline
        mock_kg_manager = MagicMock(spec=KGManager)
        mock_core_memory = MagicMock(spec=CoreMemory)
        mock_wave_mechanics = MagicMock(spec=WaveMechanics)
        mock_cellular_world = MagicMock(spec=World)
        mock_core_memory.get_unasked_hypotheses.return_value = []

        pipeline = CognitionPipeline(
            kg_manager=mock_kg_manager,
            core_memory=mock_core_memory,
            wave_mechanics=mock_wave_mechanics,
            cellular_world=mock_cellular_world
        )
        # Mock the successor of the command handler
        pipeline.entry_handler._successor._successor = MagicMock()

        response, _ = pipeline.process_message("calculate: 5 * 5")
        self.assertEqual(response['text'], "계산 결과는 25 입니다.")

if __name__ == '__main__':
    unittest.main()
