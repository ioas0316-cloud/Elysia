import unittest
from unittest.mock import patch
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.inquisitive_mind import InquisitiveMind

class TestInquisitiveMind(unittest.TestCase):

    def setUp(self):
        """Set up for the tests."""
        self.inquisitive_mind = InquisitiveMind()

    @patch('Project_Sophia.inquisitive_mind.generate_text')
    def test_ask_external_llm_success(self, mock_generate_text):
        """Tests the ask_external_llm method for a successful response."""
        mock_response = "A large language model."
        mock_generate_text.return_value = mock_response

        response = self.inquisitive_mind.ask_external_llm("LLM")

        expected_response = f"'LLM'에 대해 외부에서 이런 정보를 찾았어요: \"{mock_response}\". 이 정보가 정확한가요?"
        self.assertEqual(response, expected_response)

    @patch('Project_Sophia.inquisitive_mind.generate_text')
    def test_ask_external_llm_failure(self, mock_generate_text):
        """Tests the ask_external_llm method for a failure response."""
        mock_generate_text.side_effect = Exception("API Error")

        response = self.inquisitive_mind.ask_external_llm("LLM")

        expected_response = "I tried to look that up, but I encountered an error."
        self.assertEqual(response, expected_response)

if __name__ == '__main__':
    unittest.main()
