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

        self.kg_manager = KGManager(filepath=self.test_kg_path)
        self.core_memory = CoreMemory(file_path=self.test_memory_path)

        # We need real components for this integration test
        from Project_Elysia.memory_weaver import MemoryWeaver
        self.memory_weaver = MemoryWeaver(self.core_memory, self.kg_manager)

        # Import Guardian after setting up paths
        from Project_Elysia.guardian import Guardian
        self.guardian = Guardian()

        # Inject our real components into the Guardian instance
        self.guardian.kg_manager = self.kg_manager
        self.guardian.core_memory = self.core_memory
        self.guardian.memory_weaver = self.memory_weaver
        # Mock components that are not under test to isolate behavior
        self.guardian.daemon = MagicMock()
        self.guardian.exploration_cortex = MagicMock()
        self.guardian.web_search_cortex = MagicMock()
        self.guardian.knowledge_distiller = MagicMock()


    def tearDown(self):
        """Clean up the test files after each test."""
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)
        if os.path.exists(self.test_memory_path):
            os.remove(self.test_memory_path)

    def test_guardian_idle_cycle_triggers_reflection_and_creates_hypothesis(self):
        """
        Verify that the Guardian's idle cycle triggers the self-reflection process
        (via MemoryWeaver's volatile thought weaving), leading to the creation of a
        new high-confidence hypothesis in CoreMemory.
        """
        # 1. Set up the KG with some initial data (optional but good practice)
        self.kg_manager.add_node("AI")
        self.kg_manager.add_node("consciousness")
        self.kg_manager.add_node("tool")
        self.kg_manager.save()

        # 2. Add volatile thought fragments designed to create a high-confidence link
        self.core_memory.add_volatile_memory_fragment(["AI", "consciousness", "tool"])
        self.core_memory.add_volatile_memory_fragment(["AI", "consciousness", "ethics"])
        self.core_memory.add_volatile_memory_fragment(["AI", "consciousness", "future"])
        self.core_memory.add_volatile_memory_fragment(["AI", "tool", "development"])

        # 3. Set the Guardian's state to be ready for an idle cycle that triggers memory weaving
        from Project_Elysia.guardian import ElysiaState
        self.guardian.current_state = ElysiaState.IDLE
        self.guardian.treasure_is_safe = True
        self.guardian.last_learning_time = 0
        self.guardian.last_activity_time = time.time() - (self.guardian.time_to_idle + 1)
        self.guardian.daemon.is_alive = True

        # 4. Run the Guardian's idle cycle, which should trigger learning
        self.guardian.run_idle_cycle()

        # 5. Assert that a new hypothesis has been created in CoreMemory
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
