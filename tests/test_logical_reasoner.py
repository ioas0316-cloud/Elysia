import unittest
import os
import sys
from unittest.mock import MagicMock

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.logical_reasoner import LogicalReasoner
from tools.kg_manager import KGManager
from Project_Sophia.cognition_pipeline import CognitionPipeline

class TestLogicalReasonerIntegration(unittest.TestCase):
    """
    Test the integration of LogicalReasoner within the CognitionPipeline.
    """
    def setUp(self):
        """
        Setup a pipeline with a reasoner that uses an in-memory KG.
        """
        self.kg_manager_instance = KGManager()
        self.kg_manager_instance.kg = {"nodes": [], "edges": []}

        self.kg_manager_instance.add_node("소크라테스", properties={"description": "고대 그리스의 철학자"})
        self.kg_manager_instance.add_node("인간")
        self.kg_manager_instance.add_or_update_edge("소크라테스", "인간", "is_a")

        self.pipeline = CognitionPipeline()
        self.reasoner_with_test_kg = LogicalReasoner(kg_manager=self.kg_manager_instance)
        self.pipeline.reasoner = self.reasoner_with_test_kg
        self.pipeline.api_available = False

    def test_reasoning_and_response(self):
        """
        Tests if the pipeline, using the configured reasoner, produces the correct response.
        """
        message = "소크라테스에 대해 알려줘"
        response, _ = self.pipeline.process_message(message)

        # The stable reasoner only lists relationships.
        self.assertIn("'소크라테스'은(는) '인간'와(과) 'is_a' 관계를 가집니다.", response['text'])

        print("\n--- 테스트 통과 ---")
        print(f"입력: '{message}'")
        print(f"응답: {response}")
        print("-------------------")

if __name__ == '__main__':
    unittest.main()
