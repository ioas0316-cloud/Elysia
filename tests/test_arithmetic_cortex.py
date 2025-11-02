import unittest
from unittest.mock import patch
import os
import sys

# Add the project root to the Python path to resolve module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.arithmetic_cortex import ArithmeticCortex
from Project_Sophia.cognition_pipeline import CognitionPipeline

class TestArithmeticCortex(unittest.TestCase):

    def setUp(self):
        """Set up for the tests."""
        self.cortex = ArithmeticCortex()
        self.pipeline = CognitionPipeline()

    def test_evaluate_simple_expressions(self):
        """Test evaluation of simple arithmetic expressions."""
        self.assertEqual(self.cortex.evaluate("2 * 3"), 6)
        self.assertEqual(self.cortex.evaluate("10 / 2"), 5)
        self.assertEqual(self.cortex.evaluate("5 - 1"), 4)
        self.assertEqual(self.cortex.evaluate("1.5 + 2.5"), 4.0)

    def test_evaluate_with_parentheses(self):
        """Test order of operations with parentheses."""
        self.assertEqual(self.cortex.evaluate("(2 + 3) * 4"), 20)
        self.assertEqual(self.cortex.evaluate("100 / (10 * 2)"), 5)

    def test_evaluate_unsafe_expression(self):
        """Test that unsafe expressions are not evaluated."""
        self.assertIsNone(self.cortex.evaluate("__import__('os').system('echo unsafe')"))
        self.assertIsNone(self.cortex.evaluate("a + b"))

    def test_verify_truth(self):
        """Test the improved verify_truth method."""
        self.assertTrue(self.cortex.verify_truth("2 * 3 = 6"))
        self.assertFalse(self.cortex.verify_truth("10 / 2 = 4"))
        self.assertTrue(self.cortex.verify_truth("(2 + 3) * 4 = 20"))

    @patch('Project_Sophia.local_llm_cortex.LocalLLMCortex.generate_response', return_value="계산 결과는 15 입니다.")
    def test_pipeline_integration_question(self, mock_local_llm_response):
        """Test the pipeline integration for calculation questions."""
        # Clean up potentially corrupted memory file before test
        main_memory_path = 'Elysia_Input_Sanctum/elysia_core_memory.json'
        if os.path.exists(main_memory_path):
            os.remove(main_memory_path)
        
        pipeline = CognitionPipeline() # Re-initialize with clean memory
        response, _ = pipeline.process_message("5 * 3는?")
        self.assertEqual(response['text'], "계산 결과는 15 입니다.")

    @patch('Project_Sophia.local_llm_cortex.LocalLLMCortex.generate_response', return_value="계산 결과는 25 입니다.")
    def test_pipeline_integration_command(self, mock_local_llm_response):
        """Test the pipeline integration for calculation commands."""
        # Clean up potentially corrupted memory file before test
        main_memory_path = 'Elysia_Input_Sanctum/elysia_core_memory.json'
        if os.path.exists(main_memory_path):
            os.remove(main_memory_path)

        pipeline = CognitionPipeline() # Re-initialize with clean memory
        response, _ = pipeline.process_message("계산해줘: 100 / 4")
        self.assertEqual(response['text'], "계산 결과는 25 입니다.")

if __name__ == '__main__':
    unittest.main()
