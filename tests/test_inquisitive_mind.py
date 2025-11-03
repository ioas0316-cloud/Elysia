import unittest
from unittest.mock import patch, MagicMock
from Project_Sophia.inquisitive_mind import InquisitiveMind

class TestInquisitiveMind(unittest.TestCase):
    def test_ask_external_llm_success(self):
        mock_cortex = MagicMock()
        mock_cortex.generate_response.return_value = "Test knowledge"
        mind = InquisitiveMind(llm_cortex=mock_cortex)
        response, success = mind.ask_external_llm("anything")
        self.assertTrue(success)
        self.assertIn("Test knowledge", response)

    def test_ask_external_llm_empty_response(self):
        mock_cortex = MagicMock()
        mock_cortex.generate_response.return_value = ""
        mind = InquisitiveMind(llm_cortex=mock_cortex)
        response, success = mind.ask_external_llm("anything")
        self.assertFalse(success)
        self.assertEqual(response, "I tried to find out, but I was unable to get a clear answer.")

    def test_ask_external_llm_api_error(self):
        mock_cortex = MagicMock()
        mock_cortex.generate_response.side_effect = Exception("API Error")
        mind = InquisitiveMind(llm_cortex=mock_cortex)
        response, success = mind.ask_external_llm("anything")
        self.assertFalse(success)
        self.assertEqual(response, "I tried to find out, but I was unable to get a clear answer.")

if __name__ == '__main__':
    unittest.main()
