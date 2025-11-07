import unittest
from unittest.mock import MagicMock

from Project_Sophia.exploration_cortex import ExplorationCortex
from nano_core.bus import MessageBus
from tools.kg_manager import KGManager

class TestExplorationCortex(unittest.TestCase):

    def setUp(self):
        """Set up a mock KGManager and a real MessageBus for each test."""
        self.bus = MessageBus()
        self.mock_kg = MagicMock(spec=KGManager)
        self.explorer = ExplorationCortex(self.mock_kg, self.bus)

    def test_explore_and_hypothesize_finds_gap(self):
        """
        Test that the cortex can find a 2-step gap (A->B->C) and propose a link between A and C.
        """
        # Setup: A simple graph A->B, B->C. The gap is A to C.
        self.mock_kg.kg = {
            'nodes': [
                {'id': 'A', 'mass': 2.0},  # Interesting node
                {'id': 'B'},
                {'id': 'C'}
            ],
            'edges': [
                {'source': 'A', 'target': 'B', 'relation': 'related_to'},
                {'source': 'B', 'target': 'C', 'relation': 'related_to'}
            ]
        }

        # To make the test deterministic, we guide the random walk: A -> B -> C
        def controlled_choice(sequence):
            if 'A' in sequence and len(sequence) == 1: # Interesting nodes
                return 'A'
            if 'B' in sequence and 'C' in sequence: # Neighbors of A in the second test
                 return 'B'
            if 'A' in sequence and 'C' in sequence: # Neighbors of B
                return 'C'
            # Fallback for the first test's neighbor selection
            if 'B' in sequence:
                return 'B'
            return sequence[0] if sequence else None

        with unittest.mock.patch('random.choice', side_effect=controlled_choice):
             self.explorer.explore_and_hypothesize(num_hypotheses=1)

        # Verification: A 'validate' message for (A, C) should be on the bus
        self.assertFalse(self.bus.empty(), "Message bus should not be empty after exploration.")

        hypothesis_msg = self.bus.get_next()
        self.assertIsNotNone(hypothesis_msg)

        self.assertEqual(hypothesis_msg.verb, 'validate')
        self.assertEqual(hypothesis_msg.src, 'ExplorationCortex')
        self.assertAlmostEqual(hypothesis_msg.strength, 0.1)

        slots = hypothesis_msg.slots
        # The relationship could be (A, C) or (C, A), so we check both possibilities
        subj_obj = {slots.get('subject'), slots.get('object')}
        self.assertIn('A', subj_obj)
        self.assertIn('C', subj_obj)
        self.assertEqual(slots.get('relation'), 'related_to')

    def test_does_not_propose_existing_links(self):
        """Test that the cortex does not propose a link if one already exists."""
        # Setup: A->B, B->C, and A->C already exists
        self.mock_kg.kg = {
            'nodes': [{'id': 'A', 'mass': 2.0}, {'id': 'B'}, {'id': 'C'}],
            'edges': [
                {'source': 'A', 'target': 'B', 'relation': 'related_to'},
                {'source': 'B', 'target': 'C', 'relation': 'related_to'},
                {'source': 'A', 'target': 'C', 'relation': 'related_to'} # Direct link exists
            ]
        }

        # To make the test deterministic, we guide the random walk: A -> B -> C
        def controlled_choice(sequence):
            if 'A' in sequence and len(sequence) == 1: # Interesting nodes
                return 'A'
            # Neighbors of A is just B in this specific graph setup for this test
            if 'B' in sequence and 'C' not in sequence:
                return 'B'
            if 'A' in sequence and 'C' in sequence: # Neighbors of B
                return 'C'
            return sequence[0] if sequence else None

        with unittest.mock.patch('random.choice', side_effect=controlled_choice):
            self.explorer.explore_and_hypothesize(num_hypotheses=1)

        # Verification: The bus should be empty
        self.assertTrue(self.bus.empty())

if __name__ == '__main__':
    unittest.main()
