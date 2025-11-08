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
            "'햇빛'은(는) '식물 성장'을(를) 유발할 수 있습니다." in t.content and
            t.confidence > 0.9
            for t in thoughts
        )
        self.assertTrue(static_thought_found, "Should deduce a static fact from the KG.")

        # Check dynamic thought from simulation
        # The exact text might vary, so we check for key elements.
        dynamic_thought_found = any(
            t.source == 'flesh' and
            "햇빛" in t.content and
            "식물 성장" in t.content and
            "활성화" in t.content and
            t.confidence < 0.8 and
            t.energy > 1.0
            for t in thoughts
        )
        self.assertTrue(dynamic_thought_found, "Should produce a dynamic insight from simulation.")

        # Verify all elements are Thought instances
        for item in thoughts:
            self.assertIsInstance(item, Thought)

if __name__ == '__main__':
    unittest.main()
