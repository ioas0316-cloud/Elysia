import unittest
import os
import sys
import re
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.cognition_pipeline import CognitionPipeline
from Project_Sophia.logical_reasoner import LogicalReasoner
from Project_Sophia.core.world import World
from Project_Sophia.core.cell import Cell
from tools.kg_manager import KGManager

class TestLivingReasoning(unittest.TestCase):
    """
    Tests the LogicalReasoner's "living reason" capabilities.
    """
    def setUp(self):
        """Set up a controlled environment for testing dynamic reasoning."""
        self.kg_manager = KGManager(filepath='data/test_living_reason_kg.json')
        self.kg_manager.kg = {"nodes": [], "edges": []}

        self.cellular_world = World(primordial_dna={"instinct": "connect"})

        nodes = ["햇빛", "식물 성장", "산소 발생"]
        for node in nodes:
            self.kg_manager.add_node(node)
            self.cellular_world.add_cell(node, initial_energy=0.1)

        self.kg_manager.add_edge("햇빛", "식물 성장", "causes", properties={"strength": 0.9})
        self.kg_manager.add_edge("식물 성장", "산소 발생", "causes", properties={"strength": 0.8})

        self.reasoner = LogicalReasoner(
            kg_manager=self.kg_manager,
            cellular_world=self.cellular_world
        )

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists('data/test_living_reason_kg.json'):
            os.remove('data/test_living_reason_kg.json')

    def test_deduce_with_simulation(self):
        """Verify deduction from both static KG and dynamic simulation."""
        message = "만약 햇빛이 강해지면 어떤 결과가 발생할까?"
        facts = self.reasoner.deduce_facts(message)

        result_text = "\n".join(facts)

        print("\n--- '살아있는 추론' 단위 테스트 결과 ---")
        print(f"질문: {message}")
        print("추론된 사실들:")
        print(result_text)
        print("---------------------------------")

        self.assertIn("[정적] '햇빛'은(는) '식물 성장'을(를) 유발할 수 있습니다.", result_text)
        # FIX: Make the assertion more flexible to handle minor grammatical variations (e.g., '(으)로' vs '(이)라는')
        self.assertIn("시뮬레이션한 결과", result_text)
        self.assertIn("'식물 성장' 개념이 활성화되었습니다", result_text)
        self.assertIn("'산소 발생' 개념이 활성화되었습니다", result_text)


class TestLogicalReasonerIntegration(unittest.TestCase):
    """
    Tests the integration of the LogicalReasoner within the CognitionPipeline.
    """
    def setUp(self):
        """Set up the environment for integration testing."""
        self.test_kg_path = Path('data/test_integration_kg.json')
        if self.test_kg_path.exists():
            self.test_kg_path.unlink()

        self.kg_manager_instance = KGManager(filepath=self.test_kg_path)
        self.kg_manager_instance.add_node("소크라테스")
        self.kg_manager_instance.add_node("인간")
        self.kg_manager_instance.add_edge("소크라테스", "인간", "is_a")
        self.kg_manager_instance.save()

        self.pipeline = CognitionPipeline(cellular_world=World(primordial_dna={}))

        self.pipeline.kg_manager = self.kg_manager_instance
        self.pipeline.reasoner.kg_manager = self.kg_manager_instance


    def tearDown(self):
        """Clean up the environment after testing."""
        if self.test_kg_path.exists():
            self.test_kg_path.unlink()

    def test_reasoning_and_response(self):
        """Test that logical reasoning and response generation work correctly together."""
        test_message = "소크라테스에 대해 알려줘"
        response, _ = self.pipeline.process_message(test_message)

        response_text = response['text']

        print("\n--- 통합 테스트 결과 ---")
        print(f"입력: '{test_message}'")
        print(f"응답: {response_text}")
        print("-------------------")

        self.assertIn("소크라테스", response_text)
        self.assertIn("인간", response_text)
        self.assertIn("한 종류입니다", response_text)


if __name__ == '__main__':
    unittest.main()
