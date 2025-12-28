import unittest
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Core.Foundation.logical_reasoner import LogicalReasoner
from tools.kg_manager import KGManager

class TestCausalReasoning(unittest.TestCase):

    def setUp(self):
        """Set up a fresh reasoner and a test KG for each test."""
        self.test_kg_path = Path('data/test_causal_kg.json')
        if self.test_kg_path.exists():
            self.test_kg_path.unlink()

        self.test_kg_manager = KGManager(filepath=self.test_kg_path)
        self.reasoner = LogicalReasoner(kg_manager=self.test_kg_manager)

        # Set up a test KG with various relationships
        self.reasoner.kg_manager.add_edge("햇빛", "식물 성장", "causes")
        self.reasoner.kg_manager.add_edge("수분", "식물 성장", "causes")
        self.reasoner.kg_manager.add_edge("식물 성장", "산소 발생", "causes")
        self.reasoner.kg_manager.add_node("소크라테스")
        self.reasoner.kg_manager.add_node("인간")
        self.reasoner.kg_manager.add_edge("소크라테스", "인간", "is_a")

    def tearDown(self):
        pass # No file cleanup needed as KG is in-memory for tests

    def test_deduce_causes(self):
        """Tests if the reasoner correctly identifies causes."""
        thoughts = self.reasoner.deduce_facts("식물 성장")
        fact_contents = [t.content for t in thoughts]

        expected_causes = [
            "'햇빛'은(는) '식물 성장'의 원인이 될 수 있습니다.",
            "'수분'은(는) '식물 성장'의 원인이 될 수 있습니다."
        ]
        for cause in expected_causes:
            self.assertIn(cause, fact_contents)

    def test_deduce_effects(self):
        """Tests if the reasoner correctly identifies effects."""
        thoughts = self.reasoner.deduce_facts("식물 성장")
        fact_contents = [t.content for t in thoughts]
        expected_effect = "'식물 성장'은(는) '산소 발생'을(를) 유발할 수 있습니다."
        self.assertIn(expected_effect, fact_contents)

    def test_deduce_general_non_causal_relationship(self):
        """Tests if the reasoner handles general, non-causal queries correctly."""
        thoughts = self.reasoner.deduce_facts("소크라테스")
        fact_contents = [t.content for t in thoughts]
        expected_fact = "'소크라테스'은(는) '인간'의 한 종류입니다."
        self.assertIn(expected_fact, fact_contents)

    def test_ambiguous_query_returns_all_related_facts(self):
        """
        Tests if a general query about an entity returns all related facts (causes, effects, etc.).
        """
        thoughts = self.reasoner.deduce_facts("식물 성장")
        fact_contents = [t.content for t in thoughts]

        expected_facts = [
            "'햇빛'은(는) '식물 성장'의 원인이 될 수 있습니다.",
            "'수분'은(는) '식물 성장'의 원인이 될 수 있습니다.",
            "'식물 성장'은(는) '산소 발생'을(를) 유발할 수 있습니다."
        ]
        self.assertCountEqual(fact_contents, expected_facts)


if __name__ == '__main__':
    unittest.main()
