
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.S1_Body.L5_Mental.Reasoning_Core.LLM.local_cortex import LocalCortex

class TestLocalCortexEmbedding(unittest.TestCase):
    @patch('requests.post')
    @patch('requests.get')
    def test_embed_success(self, mock_get, mock_post):
        # Mock connection check
        mock_get.return_value.status_code = 200

        # Mock embedding response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response

        cortex = LocalCortex()
        vector = cortex.embed("Test Concept")

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        mock_post.assert_called_with(
            "http://localhost:11434/api/embeddings",
            json={"model": "llama3:latest", "prompt": "Test Concept"}
        )

    @patch('requests.get')
    def test_embed_inactive(self, mock_get):
        # Mock connection fail
        mock_get.side_effect = Exception("Connection Refused")

        cortex = LocalCortex()
        vector = cortex.embed("Test")

        self.assertEqual(len(vector), 768)
        self.assertEqual(vector[0], 0.0)

if __name__ == "__main__":
    unittest.main()
