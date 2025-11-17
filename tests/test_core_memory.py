import unittest
from unittest.mock import patch, MagicMock
import os
import json
from datetime import datetime

# To allow the test to run from the root directory, we need to adjust the path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Elysia.core_memory import CoreMemory, Memory, EmotionalState

class TestCoreMemory(unittest.TestCase):

    def setUp(self):
        """Set up a fresh CoreMemory instance for each test."""
        # Use in-memory mode and a small capacity for easier testing
        self.memory = CoreMemory(file_path=None, memory_capacity=5)

    def test_01_add_experience_within_capacity(self):
        """Test that experiences are added correctly when within capacity."""
        self.assertEqual(len(self.memory.get_experiences()), 0)
        mem1 = Memory(timestamp=datetime.now().isoformat(), content="First memory")
        self.memory.add_experience(mem1)
        self.assertEqual(len(self.memory.get_experiences()), 1)
        self.assertEqual(self.memory.get_experiences()[-1].content, "First memory")

    def test_02_ring_buffer_circulation(self):
        """Test the circular buffer functionality, ensuring oldest memories are replaced."""
        # Fill the memory to capacity
        for i in range(5):
            self.memory.add_experience(Memory(timestamp=datetime.now().isoformat(), content=f"Memory {i}"))

        self.assertEqual(len(self.memory.get_experiences()), 5)
        self.assertEqual(self.memory.get_experiences()[0].content, "Memory 0")

        # Add one more experience, which should push out the oldest one ("Memory 0")
        new_mem = Memory(timestamp=datetime.now().isoformat(), content="Memory 5")
        self.memory.add_experience(new_mem)

        experiences = self.memory.get_experiences()
        self.assertEqual(len(experiences), 5)
        self.assertEqual(experiences[0].content, "Memory 1")
        self.assertEqual(experiences[-1].content, "Memory 5")

    @patch.object(CoreMemory, 'distill_memory')
    def test_03_distillation_is_triggered(self, mock_distill):
        """Test that the distill_memory method is called when the deque is full."""
        # Fill the memory to capacity
        for i in range(5):
            self.memory.add_experience(Memory(timestamp=datetime.now().isoformat(), content=f"Memory {i}"))

        mock_distill.assert_not_called()

        # This call should trigger distillation
        self.memory.add_experience(Memory(timestamp=datetime.now().isoformat(), content="Memory 5"))

        mock_distill.assert_called_once()
        # Check that it was called with the data of the oldest memory ("Memory 0")
        self.assertEqual(mock_distill.call_args[0][0]['content'], "Memory 0")

    def test_04_save_and_load_preserves_deque(self):
        """Test that saving and loading memory correctly handles the deque."""
        test_file = 'test_memory.json'

        # 1. Setup memory with a file path and add data
        file_based_memory = CoreMemory(file_path=test_file, memory_capacity=3)

        for i in range(4): # Add 4 memories, so one is pushed out
            file_based_memory.add_experience(Memory(timestamp=str(i), content=f"Memory {i}"))

        # Verify state before saving
        self.assertEqual(len(file_based_memory.get_experiences()), 3)
        self.assertEqual(file_based_memory.get_experiences()[0].content, "Memory 1")

        # 2. The _save_memory call is internal to add_experience. Now, load it into a new instance.
        new_memory_instance = CoreMemory(file_path=test_file)

        # 3. Verify the new instance has the correct state
        self.assertIsInstance(new_memory_instance.data['experiences'], type(self.memory.data['experiences'])) # Check it's a deque

        loaded_experiences = new_memory_instance.get_experiences()
        self.assertEqual(len(loaded_experiences), 3)
        self.assertEqual(loaded_experiences[0].content, "Memory 1")
        self.assertEqual(loaded_experiences[-1].content, "Memory 3")

        # Clean up the test file
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == '__main__':
    unittest.main()
