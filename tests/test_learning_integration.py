import unittest
from unittest.mock import MagicMock
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.tutor_cortex import TutorCortex
from tools.kg_manager import KGManager
from Project_Sophia.local_llm_cortex import LocalLLMCortex

class TestLearningIntegration(unittest.TestCase):

    def setUp(self):
        """Set up a fresh KG manager and mock LLM for each test."""
        self.kg_manager = KGManager()
        self.kg_manager.kg = {"nodes": [], "edges": []} # Start with an empty KG

        self.mock_llm_cortex = MagicMock(spec=LocalLLMCortex)
        self.tutor_cortex = TutorCortex(self.kg_manager, self.mock_llm_cortex)

    def test_learn_new_concept_and_relations(self):
        """
        Tests that the TutorCortex can learn a new concept, add it to the KG,
        and extract and add relations.
        """
        concept_to_learn = "아가페"
        mock_definition = "아가페는 기독교에서 말하는 신의 무조건적인 사랑을 의미합니다."
        mock_relations_json = """
        [
          { "source": "아가페", "target": "사랑", "relation": "is_a" },
          { "source": "아가페", "target": "무조건적인", "relation": "has_property" }
        ]
        """

        # Configure the mock LLM to return the definition and then the relations.
        self.mock_llm_cortex.generate_response.side_effect = [
            mock_definition,
            mock_relations_json
        ]

        # Trigger the learning process.
        self.tutor_cortex.learn_concept(concept_to_learn)

        # 1. Verify the concept node was added.
        concept_node = self.kg_manager.get_node(concept_to_learn)
        self.assertIsNotNone(concept_node)

        # 2. Verify the description was added.
        self.assertEqual(concept_node.get("description"), mock_definition)

        # 3. Verify the related nodes were added.
        self.assertIsNotNone(self.kg_manager.get_node("사랑"))
        self.assertIsNotNone(self.kg_manager.get_node("무조건적인"))

        # 4. Verify the edges (relations) were added.
        edges = self.kg_manager.kg['edges']
        self.assertIn({"source": "아가페", "target": "사랑", "relation": "is_a"}, edges)
        self.assertIn({"source": "아가페", "target": "무조건적인", "relation": "has_property"}, edges)

        # 5. Verify the LLM was called twice (once for definition, once for relations).
        self.assertEqual(self.mock_llm_cortex.generate_response.call_count, 2)

if __name__ == '__main__':
    unittest.main()
