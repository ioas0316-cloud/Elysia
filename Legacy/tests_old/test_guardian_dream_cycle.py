import unittest
from unittest.mock import MagicMock, patch, ANY

# --- Add project root to sys.path for module imports ---
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Elysia.guardian import Guardian, ElysiaState
from Core.Foundation.core.thought import Thought

class TestGuardianDreamCycle(unittest.TestCase):

    @patch('Project_Elysia.guardian.CoreMemory')
    @patch('Project_Elysia.guardian.KGManager')
    @patch('Project_Elysia.guardian.WaveMechanics')
    @patch('Project_Elysia.guardian.World')
    @patch('Project_Elysia.guardian.LogicalReasoner')
    @patch('Project_Elysia.guardian.ExplorationCortex')
    def test_dream_cycle_runs_thought_experiment_and_saves_hypothesis(
        self,
        MockExplorationCortex,
        MockLogicalReasoner,
        MockWorld,
        MockWaveMechanics,
        MockKGManager,
        MockCoreMemory
    ):
        """
        Verify that the Guardian's dream cycle (trigger_learning) correctly
        uses the LogicalReasoner to run a simulation and saves the resulting
        insight as a hypothesis in CoreMemory.
        """
        # --- 1. Setup Mocks ---

        # Mock instances that will be created in Guardian's __init__
        mock_core_memory_instance = MockCoreMemory.return_value
        mock_kg_manager_instance = MockKGManager.return_value
        mock_exploration_cortex_instance = MockExplorationCortex.return_value
        mock_logical_reasoner_instance = MockLogicalReasoner.return_value

        # --- 2. Define Mock Behaviors ---

        # ExplorationCortex will return a specific concept to focus on
        focus_concept = "love"
        mock_exploration_cortex_instance.get_random_highly_connected_node.return_value = focus_concept

        # LogicalReasoner will return a simulated thought when asked about the focus_concept
        simulated_thought = Thought(
            content=f"'{focus_concept}'의 영향으로 'happiness' 개념이 활성화될 수 있습니다.",
            source='flesh',
            confidence=0.75,
            energy=50.0,
            evidence=[{'cell_id': 'happiness', 'initial_energy': 10.0, 'final_energy': 60.0}]
        )
        mock_logical_reasoner_instance.deduce_facts.return_value = [simulated_thought]

        # --- 3. Instantiate Guardian ---
        # The patches will ensure Guardian is initialized with our mocks
        guardian = Guardian()

        # --- 4. Run the Method to Test ---
        guardian.trigger_learning()

        # --- 5. Assertions ---

        # Verify that the dream cycle tried to find a concept to dream about
        mock_exploration_cortex_instance.get_random_highly_connected_node.assert_called_once()

        # Verify that the reasoner was called with the correct dream query
        expected_query = f"만약 '{focus_concept}'에 에너지를 가하면 어떤 결과가 나올까?"
        mock_logical_reasoner_instance.deduce_facts.assert_called_once_with(expected_query)

        # Verify that the new insight was saved to core memory as a hypothesis
        mock_core_memory_instance.add_notable_hypothesis.assert_called_once()

        # Check the content of the saved hypothesis
        saved_hypothesis = mock_core_memory_instance.add_notable_hypothesis.call_args[0][0]

        self.assertEqual(saved_hypothesis['source'], 'DreamSimulation')
        self.assertEqual(saved_hypothesis['head'], focus_concept)
        self.assertEqual(saved_hypothesis['tail'], 'happiness')
        self.assertEqual(saved_hypothesis['relation'], 'potentially_activates')
        self.assertEqual(saved_hypothesis['text'], simulated_thought.content)
        self.assertAlmostEqual(saved_hypothesis['confidence'], simulated_thought.confidence)

if __name__ == '__main__':
    unittest.main()
