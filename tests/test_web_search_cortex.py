
import unittest
from unittest.mock import MagicMock

# Import the class to be tested
from Project_Sophia.web_search_cortex import WebSearchCortex

class TestWebSearchCortex(unittest.TestCase):

    def test_search_filters_harmful_content(self):
        """
        Tests that the search method correctly filters out harmful content
        using the SafetyGuardian and an injected search tool.
        """
        # Create a mock search tool
        mock_search_tool = MagicMock()
        mock_search_tool.return_value = [
            {"title": "Safe Content", "snippet": "This is a safe snippet."},
            {"title": "Harmful Content", "snippet": "This snippet contains violence."},
            {"title": "Another Safe One", "snippet": "More safe content here."},
        ]

        # Inject the mock tool into the WebSearchCortex
        web_search_cortex = WebSearchCortex(search_tool=mock_search_tool)
        query = "test query"

        filtered_results = web_search_cortex.search(query)

        # Verify the mock tool was called correctly
        mock_search_tool.assert_called_with(query=query)

        # Verify the filtering logic
        self.assertEqual(len(filtered_results), 2)
        self.assertEqual(filtered_results[0]['title'], "Safe Content")
        self.assertEqual(filtered_results[1]['title'], "Another Safe One")

        for result in filtered_results:
            self.assertNotIn("violence", result['snippet'].lower())

    def test_search_handles_errors_gracefully(self):
        """
        Tests that the search method handles exceptions from the injected search tool
        and returns an error dictionary.
        """
        # Create a mock search tool that raises an exception
        mock_search_tool = MagicMock(side_effect=Exception("API Error"))

        # Inject the mock tool
        web_search_cortex = WebSearchCortex(search_tool=mock_search_tool)
        query = "another test query"

        result = web_search_cortex.search(query)

        self.assertIn("error", result)
        self.assertTrue("Failed to execute search" in result["error"])

if __name__ == '__main__':
    unittest.main()
