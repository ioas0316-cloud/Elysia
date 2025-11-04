import unittest
import os
import sys
import json
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.logical_reasoner import LogicalReasoner

class TestCausalReasoning(unittest.TestCase):

    def setUp(self):
        """Set up a fresh reasoner and a test KG for each test."""
        self.reasoner = LogicalReasoner()

        # Override the default KG path to use a test-specific file
        self.test_kg_path = Path('data/test_causal_kg.json')
        # This is a bit of a hack; ideally, the KGManager would be injectable.
        # For now, we manually override the path it uses.
        self.reasoner.kg_manager._kg = {"nodes": [], "edges": []}

        # Set up a test KG with causal and non-causal relationships
        self.reasoner.kg_manager.add_edge("햇빛", "식물 성장", "causes", properties={"strength": 0.85})
        # Using "수분" instead of "물" to avoid substring matching issues with test queries.
        self.reasoner.kg_manager.add_edge("수분", "식물 성장", "causes", properties={"strength": 0.9, "conditions": ["적절한 온도"]})
        self.reasoner.kg_manager.add_edge("식물 성장", "산소 발생", "causes")
        self.reasoner.kg_manager.add_node("소크라테스")
        self.reasoner.kg_manager.add_node("인간")
        self.reasoner.kg_manager.add_edge("소크라테스", "인간", "is_a")
        # Note: KGManager's internal state is what's being tested, not the file saving/loading itself.

    def tearDown(self):
        """Clean up the test KG file after each test."""
        # No file is actually created in this version of the test, so no cleanup needed.
        pass

    def test_deduce_causes(self):
        """Tests if the reasoner correctly identifies and formats causes for a given entity."""
        message = "식물 성장의 원인은 무엇이야?"
        facts = self.reasoner.deduce_facts(message)

        self.assertEqual(len(facts), 2)
        expected_facts = [
            "'수분'은(는) '식물 성장'의 원인이 될 수 있습니다. (인과 강도: 0.9) (조건: 적절한 온도)",
            "'햇빛'은(는) '식물 성장'의 원인이 될 수 있습니다. (인과 강도: 0.85)"
        ]
        self.assertCountEqual(facts, expected_facts)

    def test_deduce_effects(self):
        """Tests if the reasoner correctly identifies and formats effects for a given entity."""
        message = "식물 성장의 결과는 무엇이야?"
        facts = self.reasoner.deduce_facts(message)

        self.assertEqual(len(facts), 1)
        expected_fact = "'식물 성장'은(는) '산소 발생'을(를) 유발할 수 있습니다."
        self.assertIn(expected_fact, facts)

    def test_deduce_general_non_causal_relationship(self):
        """Tests if the reasoner still handles general, non-causal queries correctly."""
        message = "소크라테스에 대해 알려줘"
        facts = self.reasoner.deduce_facts(message)

        self.assertEqual(len(facts), 1)
        expected_fact = "'소크라테스'은(는) '인간'와(과) 'is_a' 관계를 가집니다."
        self.assertIn(expected_fact, facts)

    def test_ambiguous_query_returns_all_related_facts(self):
        """Tests if a general query about an entity involved in causal chains returns both causes and effects."""
        message = "식물 성장에 대해 알려줘"
        facts = self.reasoner.deduce_facts(message)

        self.assertEqual(len(facts), 3)
        expected_facts = [
            "'수분'은(는) '식물 성장'의 원인이 될 수 있습니다. (인과 강도: 0.9) (조건: 적절한 온도)",
            "'햇빛'은(는) '식물 성장'의 원인이 될 수 있습니다. (인과 강도: 0.85)",
            "'식물 성장'은(는) '산소 발생'을(를) 유발할 수 있습니다."
        ]
        self.assertCountEqual(facts, expected_facts)


if __name__ == '__main__':
    unittest.main()
