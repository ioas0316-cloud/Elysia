import unittest
import os
import sys
import re
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Sophia.logical_reasoner import LogicalReasoner
from Project_Sophia.core.world import World
from Project_Sophia.core.cell import Cell
from tools.kg_manager import KGManager
from Project_Sophia.wave_mechanics import WaveMechanics # Added import
from Project_Elysia.value_centered_decision import VCD # Added import

class TestLivingReasoning(unittest.TestCase):
    def setUp(self):
        self.kg_manager = KGManager(filepath='data/test_living_reason_kg.json')
        self.kg_manager.kg = {"nodes": [], "edges": []} # Start fresh

        self.cellular_world = World(primordial_dna={"instinct": "connect"})
        nodes = ["햇빛", "식물 성장", "산소 발생", "love"]
        for node in nodes:
            self.kg_manager.add_node(node, properties={'embedding': [0.1]*8})
            self.cellular_world.add_cell(node, initial_energy=0.1)

        self.kg_manager.add_edge("햇빛", "식물 성장", "causes")
        self.kg_manager.add_edge("식물 성장", "산소 발생", "causes")
        self.kg_manager.save() # Save the KG to the file

        self.reasoner = LogicalReasoner(
            kg_manager=self.kg_manager,
            cellular_world=self.cellular_world
        )

    def tearDown(self):
        if os.path.exists('data/test_living_reason_kg.json'):
            os.remove('data/test_living_reason_kg.json')

    def test_deduce_with_simulation(self):
        message = "만약 햇빛이 강해지면 어떤 결과가 발생할까?"
        facts = self.reasoner.deduce_facts(message)
        result_text = "\n".join(facts)
        self.assertIn("[정적] '햇빛'은(는) '식물 성장'을(를) 유발할 수 있습니다.", result_text)


class TestLogicalReasonerIntegration(unittest.TestCase):
    def setUp(self):
        self.test_kg_path = Path('data/test_integration_kg.json')
        if self.test_kg_path.exists():
            self.test_kg_path.unlink()

        self.kg_manager_instance = KGManager(filepath=self.test_kg_path)
        world = World(primordial_dna={})

        nodes = ["소크라테스", "인간", "love"]
        for node in nodes:
            self.kg_manager_instance.add_node(node, properties={'embedding': [0.1]*8})
            world.add_cell(node)

        self.kg_manager_instance.add_edge("소크라테스", "인간", "is_a")
        self.kg_manager_instance.save() # Save the KG to the file

        self.pipeline = CognitionPipeline(cellular_world=world)
        # Re-initialize components with the saved KG
        self.pipeline.kg_manager = KGManager(filepath=self.test_kg_path)
        self.pipeline.reasoner = LogicalReasoner(kg_manager=self.pipeline.kg_manager, cellular_world=world)
        self.pipeline.wave_mechanics = WaveMechanics(self.pipeline.kg_manager)
        self.pipeline.vcd = VCD(self.pipeline.kg_manager, self.pipeline.wave_mechanics)


    def tearDown(self):
        if self.test_kg_path.exists():
            self.test_kg_path.unlink()

    def test_reasoning_and_response(self):
        test_message = "소크라테스에 대해 알려줘"
        response, _ = self.pipeline.process_message(test_message)
        response_text = response['text']
        self.assertIn("소크라테스", response_text)
        self.assertIn("인간", response_text)
        self.assertIn("한 종류입니다", response_text)


if __name__ == '__main__':
    unittest.main()
