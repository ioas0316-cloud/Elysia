import unittest
from unittest.mock import MagicMock, patch

from nano_core.bots.validator import ValidatorBot
from nano_core.bus import MessageBus
from nano_core.message import Message
from nano_core.registry import ConceptRegistry

class TestValidatorBot(unittest.TestCase):

    def setUp(self):
        """Set up a mock ConceptRegistry and a real MessageBus for each test."""
        self.bus = MessageBus()
        self.mock_registry = MagicMock(spec=ConceptRegistry)
        # We need to mock the nested kg object as well
        self.mock_registry.kg = MagicMock()
        self.validator = ValidatorBot()

    def test_handle_new_link_validation_passes(self):
        """Test that a new, non-conflicting link passes validation and posts a 'link' message."""
        # Setup: KG has no relevant edges
        self.mock_registry.kg.kg = {'edges': []}

        validate_msg = Message(
            verb='validate',
            slots={'subject': 'A', 'object': 'B', 'relation': 'supports'},
            strength=0.9
        )

        self.validator.handle(validate_msg, self.mock_registry, self.bus)

        # Verification: A 'link' message should be on the bus
        self.assertFalse(self.bus.empty())
        link_msg = self.bus.get_next()
        self.assertEqual(link_msg.verb, 'link')
        self.assertEqual(link_msg.slots['subject'], 'A')
        self.assertEqual(link_msg.slots['object'], 'B')
        self.assertEqual(link_msg.slots['relation'], 'supports')
        self.assertEqual(link_msg.strength, 0.9) # Strength should be preserved

    def test_handle_duplicate_link_is_ignored(self):
        """Test that a duplicate link is silently ignored."""
        # Setup: KG already has the exact same edge
        self.mock_registry.kg.kg = {'edges': [
            {'source': 'A', 'target': 'B', 'relation': 'supports'}
        ]}

        validate_msg = Message(
            verb='validate',
            slots={'subject': 'A', 'object': 'B', 'relation': 'supports'}
        )

        self.validator.handle(validate_msg, self.mock_registry, self.bus)

        # Verification: The bus should be empty
        self.assertTrue(self.bus.empty())

    @patch('nano_core.bots.validator.write_event')
    def test_handle_contradictory_link_is_rejected(self, mock_write_event):
        """Test that a contradictory link is rejected and a telemetry event is written."""
        # Setup: KG has a 'refutes' edge, which contradicts 'supports'
        self.mock_registry.kg.kg = {'edges': [
            {'source': 'A', 'target': 'B', 'relation': 'refutes'}
        ]}

        validate_msg = Message(
            verb='validate',
            slots={'subject': 'A', 'object': 'B', 'relation': 'supports'}
        )

        self.validator.handle(validate_msg, self.mock_registry, self.bus)

        # Verification: The bus should be empty
        self.assertTrue(self.bus.empty())

        # Verification: A 'validation.failed' event should have been written
        mock_write_event.assert_called_once_with('validation.failed', {
            'reason': 'contradiction',
            'subject': 'A',
            'object': 'B',
            'relation': 'supports',
            'conflicting_relation': 'refutes'
        })

    def test_handle_message_with_no_subject_or_object(self):
        """Test that messages missing subject or object are ignored."""
        self.mock_registry.kg.kg = {'edges': []}

        msg_no_subj = Message(verb='validate', slots={'object': 'B', 'relation': 'supports'})
        msg_no_obj = Message(verb='validate', slots={'subject': 'A', 'relation': 'supports'})

        self.validator.handle(msg_no_subj, self.mock_registry, self.bus)
        self.assertTrue(self.bus.empty())

        self.validator.handle(msg_no_obj, self.mock_registry, self.bus)
        self.assertTrue(self.bus.empty())

if __name__ == '__main__':
    unittest.main()
