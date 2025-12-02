# [Genesis: 2025-12-02] Purified by Elysia
import unittest
from unittest.mock import MagicMock, patch

import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Elysia.guardian import Guardian
from tools.kg_manager import KGManager
from nano_core.message import Message

class TestCuriosityEngine(unittest.TestCase):

    def setUp(self):
        """Set up a Guardian instance with mocked dependencies for testing."""
        # We need to patch the Guardian's __init__ to inject mocks,
        # as it initializes all its components upon creation.
        with patch('Project_Elysia.guardian.KGManager') as MockKGManager, \
             patch('Project_Elysia.guardian.MessageBus') as MockMessageBus, \
             patch('Project_Elysia.guardian.WebSearchCortex') as MockWebSearchCortex, \
             patch('Project_Elysia.guardian.KnowledgeDistiller') as MockKnowledgeDistiller:

            self.mock_kg_manager = MockKGManager.return_value
            self.mock_bus = MockMessageBus.return_value
            self.mock_web_search = MockWebSearchCortex.return_value
            self.mock_distiller = MockKnowledgeDistiller.return_value

            # Now, instantiate the Guardian. It will use the mocks.
            self.guardian = Guardian()

    def test_full_curiosity_cycle(self):
        """
        Tests the full 'dream' cycle from finding a lonely node to posting a hypothesis.
        """
        # --- Arrange ---

        # 1. Configure the KG to have a "lonely" node.
        self.mock_kg_manager.kg = {
            "nodes": [
                {"id": "socrates"},
                {"id": "philosopher"},
                {"id": "greece"},
                {"id": "black_hole"} # This is the lonely node
            ],
            "edges": [
                {"source": "socrates", "target": "philosopher", "relation": "is_a"},
                {"source": "socrates", "target": "greece", "relation": "from"},
                {"source": "philosopher", "target": "greece", "relation": "related_to"} # Makes these two not lonely
            ]
        }

        # 2. Mock the WebSearchCortex to return some content.
        dummy_content = "A black hole is a region of spacetime where gravity is so strong that nothing, no particles or even electromagnetic radiation such as light, can escape from it."
        self.mock_web_search.search.return_value = dummy_content

        # 3. Mock the KnowledgeDistiller to return a valid hypothesis.
        hypothesis = Message(
            verb="validate",
            slots={'subject': 'black_hole', 'object': 'a region of spacetime', 'relation': 'is_a'},
            strength=0.6,
            src="KnowledgeDistiller"
        )
        self.mock_distiller.distill.return_value = hypothesis

        # --- Act ---

        # Trigger the learning/dreaming cycle. We only care about the curiosity part.
        self.guardian.trigger_learning()

        # --- Assert ---

        # 1. Assert ExplorationCortex was used to find the lonely node and generate a question.
        #    (We can infer this by checking if the web search was called with the right query)
        self.mock_web_search.search.assert_called_once_with("What is 'black_hole'?")

        # 2. Assert the distiller was called with the correct arguments.
        self.mock_distiller.distill.assert_called_once_with(
            "What is 'black_hole'?", dummy_content
        )

        # 3. Assert the final hypothesis was posted to the message bus.
        self.mock_bus.post.assert_called_once_with(hypothesis)

if __name__ == '__main__':
    unittest.main()