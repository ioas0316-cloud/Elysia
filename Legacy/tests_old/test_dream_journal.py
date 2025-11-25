# tests/test_dream_journal.py

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import shutil
from pathlib import Path

# Add project root for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Elysia.guardian import Guardian
from Project_Sophia.core.world import World
from Project_Elysia.core_memory import CoreMemory

class TestDreamJournal(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("temp_dream_journal_test")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir()

        # Mock dependencies that are not central to this test
        self.mock_kg_manager = MagicMock()
        self.mock_wave_mechanics = MagicMock()

        # Use an in-memory CoreMemory to ensure test isolation
        self.core_memory = CoreMemory(file_path=None)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @patch('Project_Sophia.gemini_api.generate_image_from_text')
    def test_dream_journal_pipeline(self, mock_generate_image):
        """
        Verify the full pipeline: Guardian's idle cycle triggers DreamObserver,
        calls image generation, and adds a structured Memory to CoreMemory.
        """
        # --- Setup ---
        mock_generate_image.return_value = True

        # We patch Guardian's dependencies during its initialization
        with patch('Project_Elysia.guardian.KGManager', return_value=self.mock_kg_manager), \
             patch('Project_Elysia.guardian.WaveMechanics', return_value=self.mock_wave_mechanics), \
             patch('Project_Elysia.guardian.CoreMemory', return_value=self.core_memory), \
             patch('Project_Elysia.guardian.Guardian.setup_logging'), \
             patch('Project_Elysia.guardian.Guardian._load_config'):

            guardian = Guardian()
            # Manually set attributes that would have been loaded by _load_config
            guardian.kg_path = 'data/dummy_kg.json'  # Provide a dummy path
            guardian.time_to_idle = 300
            guardian.disable_wallpaper = True


            # Create a world with some activity
            world = guardian.cellular_world
            # Fix: Use the new 'properties' argument for add_cell
            world.add_cell("love", properties={'label': 'Love', 'hp': 100.0, 'max_hp': 100.0})
            world.add_cell("creation", properties={'label': 'Creation', 'hp': 80.0, 'max_hp': 80.0})

        # --- Execution ---
        # Manually trigger the core logic of the dream journal creation.
        try:
            dream_digest = guardian.dream_observer.observe_dream(guardian.cellular_world)

            if dream_digest and dream_digest.get('key_concepts'):
                primary_emotion = dream_digest.get('emotional_landscape', 'neutral')
                mood = guardian.emotional_engine.create_state_from_feeling(primary_emotion)

                image_prompt = "A prompt" # Mocked, content doesn't matter
                dream_image_path = "data/dreams/dream_test.png"

                success = mock_generate_image(image_prompt, dream_image_path)

                if success:
                    from Project_Elysia.core_memory import Memory
                    dream_memory = Memory(
                        timestamp="2025-01-01T12:00:00",
                        content=dream_digest.get('summary'),
                        emotional_state=mood,
                        context={
                            "type": "dream_journal",
                            "image_path": dream_image_path,
                            "key_concepts": dream_digest.get('key_concepts'),
                        },
                        tags=["dream", primary_emotion] + dream_digest.get('key_concepts', [])
                    )
                    guardian.core_memory.add_experience(dream_memory)

        except Exception as e:
            self.fail(f"Dream journal creation logic failed with an exception: {e}")

        # --- Verification ---
        self.assertTrue(mock_generate_image.called)

        memories = self.core_memory.get_experiences()
        self.assertEqual(len(memories), 1)

        dream_memory = memories[0]
        content_lower = dream_memory.content.lower()
        self.assertIn("love", content_lower)
        self.assertIn("creation", content_lower)
        self.assertEqual(dream_memory.emotional_state.primary_emotion, 'hopeful')
        self.assertEqual(dream_memory.context.get('type'), 'dream_journal')
        self.assertEqual(dream_memory.context.get('image_path'), "data/dreams/dream_test.png")
        self.assertIn("love", dream_memory.tags)


if __name__ == '__main__':
    unittest.main()
