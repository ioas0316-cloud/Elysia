# [Genesis: 2025-12-02] Purified by Elysia
import unittest
import os
import sys
from unittest.mock import patch, MagicMock
import time

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from tools.kg_manager import KGManager
from Project_Elysia.core_memory import CoreMemory
from Project_Elysia.memory_weaver import MemoryWeaver
from Project_Elysia.guardian import Guardian, ElysiaState

class TestSelfGrowthInfrastructure(unittest.TestCase):

    def setUp(self):
        """Set up a clean environment for each test."""
        self.test_kg_path = 'data/test_kg_self_growth.json'
        self.test_memory_path = 'data/test_core_memory_self_growth.json'

        # Clean up old files
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)
        if os.path.exists(self.test_memory_path):
            os.remove(self.test_memory_path)

        # These components will be injected into the Guardian instance during the test
        self.kg_manager = KGManager(filepath=self.test_kg_path)
        self.core_memory = CoreMemory(file_path=self.test_memory_path)
        self.memory_weaver = MemoryWeaver(self.core_memory, self.kg_manager)

    def tearDown(self):
        """Clean up the test files after each test."""
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)
        if os.path.exists(self.test_memory_path):
            os.remove(self.test_memory_path)

    @patch('Project_Elysia.guardian.ElysiaDaemon')
    def test_guardian_idle_cycle_triggers_reflection_and_creates_hypothesis(self, MockElysiaDaemon):
        """
        Verify that the Guardian's idle cycle triggers the self-reflection process
        (via MemoryWeaver's volatile thought weaving), leading to the creation of a
        new high-confidence hypothesis in CoreMemory.
        """
        # 1. Instantiate Guardian. The ElysiaDaemon is automatically mocked by the patch.
        guardian = Guardian()

        # 2. Inject our real, test-specific components into the Guardian instance
        guardian.kg_manager = self.kg_manager
        guardian.core_memory = self.core_memory
        guardian.memory_weaver = self.memory_weaver

        # Mock other components not under test to isolate behavior
        guardian.exploration_cortex = MagicMock()
        guardian.web_search_cortex = MagicMock()
        guardian.knowledge_distiller = MagicMock()

        # 3. Set up the KG with some initial data
        self.kg_manager.add_node("AI")
        self.kg_manager.add_node("consciousness")
        self.kg_manager.add_node("tool")
        self.kg_manager.save()

        # 4. Add volatile thought fragments designed to create a high-confidence link
        self.core_memory.add_volatile_memory_fragment(["AI", "consciousness", "tool"])
        self.core_memory.add_volatile_memory_fragment(["AI", "consciousness", "ethics"])
        self.core_memory.add_volatile_memory_fragment(["AI", "consciousness", "future"])
        self.core_memory.add_volatile_memory_fragment(["AI", "tool", "development"])

        # 5. Set the Guardian's state to be ready for an idle cycle
        guardian.current_state = ElysiaState.IDLE
        guardian.treasure_is_safe = True
        guardian.last_learning_time = 0
        guardian.last_activity_time = time.time() - (guardian.time_to_idle + 1)
        guardian.daemon = MockElysiaDaemon.return_value
        guardian.daemon.is_alive = True

        # 6. Run the Guardian's idle cycle, which should trigger learning
        guardian.run_idle_cycle()

        # 7. Assert that a new hypothesis has been created in CoreMemory
        hypotheses = self.core_memory.get_unasked_hypotheses()
        self.assertGreater(len(hypotheses), 0, "At least one notable hypothesis should have been created.")

        # Check if the specific hypothesis we expect is in the list
        found_expected_hypothesis = False
        for h in hypotheses:
            if (h['head'] == 'AI' and h['tail'] == 'consciousness') or \
               (h['head'] == 'consciousness' and h['tail'] == 'AI'):
                self.assertGreaterEqual(h['confidence'], 0.7)
                self.assertEqual(h['source'], 'MemoryWeaver_Volatile')
                found_expected_hypothesis = True
                break

        self.assertTrue(found_expected_hypothesis, "The expected hypothesis (AI <-> consciousness) was not found.")

if __name__ == '__main__':
    unittest.main()