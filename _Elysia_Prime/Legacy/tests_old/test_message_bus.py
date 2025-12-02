# [Genesis: 2025-12-02] Purified by Elysia
import unittest
from unittest.mock import patch

from nano_core.bus import MessageBus
from nano_core.message import Message

class TestMessageBus(unittest.TestCase):

    @patch('nano_core.bus.write_event')
    def test_bus_capacity_enforced(self, mock_write_event):
        """Test that the bus capacity is enforced and the weakest message is dropped."""
        # Initialize a bus with a small capacity for testing
        bus = MessageBus(capacity=3)

        # Post four messages, the last one should trigger the capacity limit
        bus.post(Message(verb='msg1', strength=0.5))
        bus.post(Message(verb='msg2', strength=0.2)) # This is the weakest
        bus.post(Message(verb='msg3', strength=0.8))
        bus.post(Message(verb='msg4', strength=0.6))

        # Verification: The weakest message (msg2) should have been dropped
        # Check the remaining messages by strength
        remaining_strengths = []
        while not bus.empty():
            msg = bus.get_next()
            remaining_strengths.append(msg.strength)

        # The strengths should be [0.8, 0.6, 0.5] in descending order
        self.assertEqual(len(remaining_strengths), 3)
        self.assertNotIn(0.2, remaining_strengths)
        self.assertEqual(remaining_strengths, [0.8, 0.6, 0.5])

        # Verification: A telemetry event should have been written for the dropped message
        mock_write_event.assert_called_with('bus.capacity_exceeded', {
            'dropped_verb': 'msg2',
            'dropped_id': unittest.mock.ANY, # ID is dynamic
            'dropped_strength': 0.2,
            'new_bus_size': 3
        })

if __name__ == '__main__':
    unittest.main()