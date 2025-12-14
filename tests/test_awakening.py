
import unittest
import logging
from unittest.mock import MagicMock
from Core.Elysia.elysia_daemon import ElysiaDaemon
from Core.Intelligence.Will.will_engine import WillEngine

class TestAwakening(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.mock_logger = MagicMock()
        self.mock_kg = MagicMock()
        self.mock_core_memory = MagicMock()
        self.mock_wave = MagicMock()
        self.mock_world = MagicMock()
        self.mock_emotion = MagicMock()
        self.mock_meta = MagicMock()

        # Setup Core Memory to avoid errors
        self.mock_core_memory.data = {}

    def test_daemon_awakens_will(self):
        """
        Verify that ElysiaDaemon initializes the WillEngine and queries it.
        """
        daemon = ElysiaDaemon(
            kg_manager=self.mock_kg,
            core_memory=self.mock_core_memory,
            wave_mechanics=self.mock_wave,
            cellular_world=self.mock_world,
            emotional_engine=self.mock_emotion,
            meta_cognition_cortex=self.mock_meta,
            logger=self.mock_logger
        )

        # Verify WillEngine is initialized
        self.assertIsInstance(daemon.will_engine, WillEngine)

        # Verify run_cycle calls get_dominant_drive
        daemon.will_engine.get_dominant_drive = MagicMock(return_value=MagicMock(priority=0.8, intent_type="TEST"))
        daemon.will_engine.process_drive = MagicMock(return_value="I have a will.")

        # Mock pipeline process
        daemon.cognition_pipeline.process_message = MagicMock(return_value=({"text": "Response"}, None))

        daemon.run_cycle()

        daemon.will_engine.get_dominant_drive.assert_called_once()
        daemon.will_engine.process_drive.assert_called_once()
        self.mock_logger.info.assert_any_call(
            "Daemon Cycle 1: Will Engine Active - I have a will."
        )

if __name__ == "__main__":
    unittest.main()
