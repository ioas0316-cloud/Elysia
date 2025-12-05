import unittest
from unittest.mock import patch
from Core.Foundation.inquisitive_mind import InquisitiveMind

class TestInquisitiveMind(unittest.TestCase):
    def test_ask_external_llm_empty_response(self):
        """
        Tests that ask_external_llm returns a user-friendly message
        when the external LLM returns an empty string.
        """
        with patch('Project_Sophia.inquisitive_mind.generate_text') as mock_generate_text:
            mock_generate_text.return_value = ""
            mind = InquisitiveMind()
            response = mind.ask_external_llm("anything")
            self.assertEqual(response, "I tried to find out, but I was unable to get a clear answer.")

if __name__ == '__main__':
    unittest.main()
