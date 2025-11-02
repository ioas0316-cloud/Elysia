import unittest
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.action_cortex import ActionCortex

class TestActionCortex(unittest.TestCase):

    def setUp(self):
        """Set up a fresh ActionCortex for each test."""
        self.action_cortex = ActionCortex()

    def test_find_best_tool_for_calculation(self):
        """
        Tests if the _find_best_tool method correctly identifies the 'calculate' tool
        for a Korean prompt.
        """
        prompt = "계산해줘: 5 * 3"
        # Access the protected method for testing purposes
        best_tool = self.action_cortex._find_best_tool(prompt)

        self.assertEqual(best_tool, "calculate")

if __name__ == '__main__':
    unittest.main()
