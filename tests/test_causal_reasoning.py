import unittest
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.logical_reasoner import LogicalReasoner
from tools.kg_manager import KGManager

class TestCausalReasoning(unittest.TestCase):
    def setUp(self):
        """Set up an in-memory knowledge graph for each test."""
        self.kg_manager_instance = KGManager()
        self.kg_manager_instance.kg = {"nodes": [], "edges": []}

        self.reasoner = LogicalReasoner(kg_manager=self.kg_manager_instance)

        # Populate with test data using English entities for robustness
        self.reasoner.kg_manager.add_node("sunlight")
        self.reasoner.kg_manager.add_node("plant_growth")
        self.reasoner.kg_manager.add_node("oxygen_production")
        self.reasoner.kg_manager.add_node("water")

        self.reasoner.kg_manager.add_or_update_edge("sunlight", "plant_growth", "causes", properties={"strength": 0.85})
        self.reasoner.kg_manager.add_or_update_edge("water", "plant_growth", "causes", properties={"strength": 0.9, "conditions": ["adequate temperature"]})
        self.reasoner.kg_manager.add_or_update_edge("plant_growth", "oxygen_production", "causes")

        self.reasoner.kg_manager.add_node("socrates")
        self.reasoner.kg_manager.add_node("human")
        self.reasoner.kg_manager.add_or_update_edge("socrates", "human", "is_a")

    def test_deduce_all_facts(self):
        """
        Tests if the simplified reasoner returns all related facts regardless of query type.
        """
        message = "What are the causes of plant_growth?"
        facts = self.reasoner.deduce_facts(message)

        # The simplified reasoner will return all facts related to 'plant_growth'
        self.assertIn("'sunlight'은(는) 'plant_growth'의 원인이 될 수 있습니다. (인과 강도: 0.85)", facts)
        self.assertIn("'water'은(는) 'plant_growth'의 원인이 될 수 있습니다. (인과 강도: 0.9) (조건: adequate temperature)", facts)
        self.assertIn("'plant_growth'은(는) 'oxygen_production'을(를) 유발할 수 있습니다.", facts)

    def test_deduce_general_non_causal_relationship(self):
        """Tests if the reasoner handles general, non-causal queries."""
        message = "Tell me about socrates"
        facts = self.reasoner.deduce_facts(message)
        self.assertIn("'socrates'은(는) 'human'와(과) 'is_a' 관계를 가집니다.", facts)

if __name__ == '__main__':
    unittest.main()
