import unittest
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.logical_reasoner import LogicalReasoner
from Project_Sophia.core.world import World
from Project_Sophia.core.thought import Thought
from tools.kg_manager import KGManager

class TestLogicalReasoner(unittest.TestCase):
    """
    Tests the refactored LogicalReasoner to ensure it produces structured Thought objects
    from both static knowledge and dynamic simulations.
    """
    def setUp(self):
        self.test_kg_path = Path('data/test_reasoner_kg.json')
        if self.test_kg_path.exists():
            self.test_kg_path.unlink()

        # 1. Setup KGManager
        self.kg_manager = KGManager(filepath=self.test_kg_path)
        self.kg_manager.add_node("햇빛", properties={'embedding': [0.1]*8})
        self.kg_manager.add_node("식물 성장", properties={'embedding': [0.2]*8})
        self.kg_manager.add_node("산소 발생", properties={'embedding': [0.3]*8})
        self.kg_manager.add_edge("햇빛", "식물 성장", "causes")
        self.kg_manager.add_edge("식물 성장", "산소 발생", "causes")
        self.kg_manager.save()

        # 2. Setup Cellular World
        self.cellular_world = World(primordial_dna={"instinct": "grow"})
        for node in self.kg_manager.kg['nodes']:
            self.cellular_world.add_cell(node['id'], initial_energy=1.0)

        # Manually add connections for predictable simulation
        cell_sun = self.cellular_world.get_cell("햇빛")
        cell_plant = self.cellular_world.get_cell("식물 성장")
        if cell_sun and cell_plant:
            cell_sun.connect(cell_plant, "energy_transfer", strength=0.5)

        # 3. Instantiate Reasoner
        self.reasoner = LogicalReasoner(
            kg_manager=self.kg_manager,
            cellular_world=self.cellular_world
        )

    def tearDown(self):
        if self.test_kg_path.exists():
            self.test_kg_path.unlink()

    def test_deduce_facts_returns_thought_objects(self):
        """
        Verify that deduce_facts returns a list of Thought objects with correct data.
        """
        message = "햇빛이 식물 성장에 미치는 영향은 무엇인가요? 그리고 그 결과는?"

        thoughts = self.reasoner.deduce_facts(message)

        self.assertIsInstance(thoughts, list)
        self.assertGreater(len(thoughts), 0, "Should produce at least one thought.")

        # Check static thought from KG
        static_thought_found = any(
            t.source == 'bone' and
            "가설: '햇빛'은(는) '식물 성장'을(를) 유발할 수 있습니다." in t.content
            for t in thoughts
        )
        self.assertTrue(static_thought_found, "Should deduce a static hypothesis from the KG.")

        # Verify all elements are Thought instances
        for item in thoughts:
            self.assertIsInstance(item, Thought)

    def test_thought_experiment_verifies_and_refutes_hypotheses(self):
        """
        Test that the 'thought experiment' correctly verifies a true hypothesis
        and refutes a false one, adjusting confidence accordingly.
        """
        # Add a contradictory edge for refutation testing
        self.kg_manager.add_edge("산소 발생", "햇빛", "causes")
        self.kg_manager.save()

        # This will create a 'meaning' cell, simulating an unexpected discovery
        cell_plant = self.cellular_world.get_cell("식물 성장")
        cell_oxygen = self.cellular_world.get_cell("산소 발생")
        if cell_plant and cell_oxygen:
            cell_plant.connect(cell_oxygen, "produces", strength=0.8)

        message = "햇빛과 산소 발생의 관계를 알려줘."
        thoughts = self.reasoner.deduce_facts(message)

        verified_thought = None
        refuted_thought = None
        emergent_thought = None

        for t in thoughts:
            if "가설: '햇빛'은(는) '식물 성장'을(를) 유발할 수 있습니다." in t.content:
                verified_thought = t
            if "가설: '산소 발생'은(는) '햇빛'의 원인이 될 수 있습니다." in t.content:
                refuted_thought = t
            if "새로운 개념이 탄생했습니다" in t.content:
                emergent_thought = t

        # 1. Test Verification
        self.assertIsNotNone(verified_thought, "Should have found the '햇빛 -> 식물 성장' hypothesis.")
        self.assertEqual(verified_thought.experiment['outcome'], 'verified')
        self.assertGreater(verified_thought.confidence, 0.8, "Confidence should increase after verification.")

        # 2. Test Refutation
        self.assertIsNotNone(refuted_thought, "Should have found the '산소 발생 -> 햇빛' hypothesis.")
        self.assertEqual(refuted_thought.experiment['outcome'], 'refuted')
        self.assertLess(refuted_thought.confidence, 0.7, "Confidence should decrease after refutation.")

        # 3. Test Emergent Insight (if a new cell was born)
        # In this test setup, it's not guaranteed a new cell is born, so this is optional
        if emergent_thought:
            self.assertEqual(emergent_thought.source, 'flesh')
            self.assertIn('emerged', emergent_thought.evidence[0])


if __name__ == '__main__':
    unittest.main()
