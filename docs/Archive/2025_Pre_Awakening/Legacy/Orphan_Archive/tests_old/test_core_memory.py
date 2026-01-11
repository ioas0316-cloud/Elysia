import unittest
from unittest.mock import patch
import os
import copy
from datetime import datetime
from collections import deque
from dataclasses import asdict

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Project_Elysia.core_memory import (
    CoreMemory,
    Memory,
    Experience,
    EssencePrinciple,
    EmotionalState,
)


class TestCoreMemory(unittest.TestCase):
    def setUp(self):
        """Set up a fresh CoreMemory instance for each test."""
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
        for i in range(5):
            self.memory.add_experience(Memory(timestamp=datetime.now().isoformat(), content=f"Memory {i}"))

        self.assertEqual(len(self.memory.get_experiences()), 5)
        self.assertEqual(self.memory.get_experiences()[0].content, "Memory 0")

        new_mem = Memory(timestamp=datetime.now().isoformat(), content="Memory 5")
        self.memory.add_experience(new_mem)

        experiences = self.memory.get_experiences()
        self.assertEqual(len(experiences), 5)
        self.assertEqual(experiences[0].content, "Memory 1")
        self.assertEqual(experiences[-1].content, "Memory 5")

    @patch.object(CoreMemory, "distill_memory")
    def test_03_distillation_is_triggered(self, mock_distill):
        """Test that the distill_memory hook is called when the deque is full."""
        for i in range(5):
            self.memory.add_experience(Memory(timestamp=datetime.now().isoformat(), content=f"Memory {i}"))

        mock_distill.assert_not_called()
        self.memory.add_experience(Memory(timestamp=datetime.now().isoformat(), content="Memory 5"))
        mock_distill.assert_called_once()
        self.assertEqual(mock_distill.call_args[0][0]["content"], "Memory 0")

    def test_04_save_and_load_preserves_deque(self):
        """Test that saving and loading memory correctly handles the deque."""
        test_file = "test_memory.json"

        file_based_memory = CoreMemory(file_path=test_file, memory_capacity=3)
        for i in range(4):
            file_based_memory.add_experience(Memory(timestamp=str(i), content=f"Memory {i}"))

        self.assertEqual(len(file_based_memory.get_experiences()), 3)
        self.assertEqual(file_based_memory.get_experiences()[0].content, "Memory 1")

        new_memory_instance = CoreMemory(file_path=test_file)
        self.assertIsInstance(new_memory_instance.data["experiences"], type(self.memory.data["experiences"]))

        loaded_experiences = new_memory_instance.get_experiences()
        self.assertEqual(len(loaded_experiences), 3)
        self.assertEqual(loaded_experiences[0].content, "Memory 1")
        self.assertEqual(loaded_experiences[-1].content, "Memory 3")

        if os.path.exists(test_file):
            os.remove(test_file)


class TestMemoryRingEngine(unittest.TestCase):
    def setUp(self):
        """Set up an in-memory MRE with small capacities for testing."""
        self.test_capacities = {"experience": 5, "identity": 3, "essence": 2}
        self.memory = CoreMemory(file_path=None, capacities=self.test_capacities)
        self.test_file = "test_mre_memory.json"
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def _create_dummy_experience(self, content: str, valence: float = 0.5) -> Experience:
        ts = datetime.now().isoformat() + f"_{content}"
        return Experience(
            timestamp=ts,
            content=content,
            type="episode",
            emotional_state=EmotionalState(valence=valence, arousal=0.5, dominance=0.1, primary_emotion="joy"),
        )

    def test_01_initialization(self):
        """Test if the MRE initializes with the correct 3-ring structure and EFP core."""
        self.assertIsInstance(self.memory.data["experience_loop"], deque)
        self.assertEqual(self.memory.data["experience_loop"].maxlen, self.test_capacities["experience"])
        self.assertIsInstance(self.memory.data["identity_loop"], deque)
        self.assertEqual(self.memory.data["identity_loop"].maxlen, self.test_capacities["identity"])
        self.assertIsInstance(self.memory.data["essence_loop"], deque)
        self.assertEqual(self.memory.data["essence_loop"].maxlen, self.test_capacities["essence"])
        self.assertEqual(self.memory.get_efp_core(), {"E": 1.0, "F": 1.0, "P": 1.0})

    def test_02_experience_loop_filling(self):
        """Test adding experiences to the Experience Loop."""
        self.memory.add_experience(self._create_dummy_experience("First experience"))
        self.assertEqual(len(self.memory.data["experience_loop"]), 1)
        self.assertEqual(self.memory.get_experiences()[-1].content, "First experience")

    def test_03_distillation_experience_to_identity(self):
        """Test that a full Experience Loop triggers distillation into the Identity Loop."""
        for i in range(self.test_capacities["experience"]):
            self.memory.add_experience(self._create_dummy_experience(f"Exp {i}"))
        self.assertEqual(len(self.memory.data["identity_loop"]), 0)
        self.memory.add_experience(self._create_dummy_experience("Exp 5, the trigger"))
        self.assertEqual(len(self.memory.data["identity_loop"]), 1)
        self.assertIn("Exp 0", self.memory.get_identity_fragments()[0].content)

    def test_04_distillation_identity_to_essence(self):
        """Test that a full Identity Loop triggers distillation into the Essence Loop."""
        for i in range(self.test_capacities["identity"]):
            self.memory._distill_experience_to_identity([asdict(self._create_dummy_experience(f"Source Exp {i}"))])
        self.assertEqual(len(self.memory.data["essence_loop"]), 0)
        initial_efp = copy.deepcopy(self.memory.get_efp_core())
        self.memory._distill_experience_to_identity([asdict(self._create_dummy_experience("Source Exp 3, the trigger"))])
        self.assertEqual(len(self.memory.data["essence_loop"]), 1)
        self.assertIn("Source Exp 0", self.memory.get_essence_principles()[0].content)
        self.assertNotEqual(initial_efp, self.memory.get_efp_core())
        self.assertGreater(self.memory.get_efp_core()["E"], initial_efp["E"])

    def test_05_efp_core_dynamics_on_essence_cycle(self):
        """Test EFP core updates when Essence Loop cycles."""
        for i in range(self.test_capacities["essence"]):
            impact = {"E": 0.1 * (i + 1), "F": 0.0, "P": 0.0}
            principle = EssencePrinciple(datetime.now().isoformat(), f"Principle {i}", "core_belief", [], impact)
            self.memory.data["essence_loop"].append(asdict(principle))
            self.memory._update_efp_core(impact)
        initial_e_value = self.memory.get_efp_core()["E"]
        self.memory._distill_identity_to_essence([{"content": "Trigger fragment", "timestamp": datetime.now().isoformat()}])
        final_e_value = self.memory.get_efp_core()["E"]
        self.assertLess(final_e_value, initial_e_value)

    def test_06_full_cycle_integration(self):
        """Test the full cycle from experience to essence and EFP update."""
        num_additions_for_one_essence = self.test_capacities["experience"] + self.test_capacities["identity"] + 1
        initial_efp = copy.deepcopy(self.memory.get_efp_core())

        for i in range(num_additions_for_one_essence):
            self.memory.add_experience(self._create_dummy_experience(f"Full cycle exp {i}"))

        self.assertEqual(len(self.memory.data["experience_loop"]), 5)
        self.assertEqual(len(self.memory.data["identity_loop"]), 3)
        self.assertEqual(len(self.memory.data["essence_loop"]), 1)
        self.assertNotEqual(self.memory.get_efp_core(), initial_efp)

    def test_07_save_and_load_mre(self):
        """Test that saving and loading the MRE preserves the state of all 3 rings and the EFP core."""
        file_memory = CoreMemory(file_path=self.test_file, capacities=self.test_capacities)
        for i in range(7):
            file_memory.add_experience(self._create_dummy_experience(f"Exp {i}"))

        self.assertEqual(len(file_memory.data["identity_loop"]), 2)
        original_efp = copy.deepcopy(file_memory.get_efp_core())

        new_memory = CoreMemory(file_path=self.test_file, capacities=self.test_capacities)
        self.assertEqual(len(new_memory.data["experience_loop"]), 5)
        self.assertEqual(new_memory.get_experiences()[0].content, "Exp 2")
        self.assertEqual(len(new_memory.data["identity_loop"]), 2)
        fragments = new_memory.get_identity_fragments()
        self.assertIn("Exp 0", fragments[0].content)
        self.assertIn("Exp 1", fragments[1].content)
        self.assertEqual(len(new_memory.data["essence_loop"]), 0)
        self.assertEqual(new_memory.get_efp_core(), original_efp)


if __name__ == "__main__":
    unittest.main()
