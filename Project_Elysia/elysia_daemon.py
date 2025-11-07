# elysia_daemon.py - Refactored to be a managed component of the Guardian

import json
import os
from datetime import datetime

# The daemon now uses the central pipeline for all cognitive tasks.
from .cognition_pipeline import CognitionPipeline

STATE_FILE = 'elysia_state.json'

class ElysiaDaemon:
    """
    ElysiaDaemon is no longer an independent process, but a class managed by the Guardian.
    It hosts the primary cognitive engine (CognitionPipeline) and manages Elysia's core state.
    """
    def __init__(self, cellular_world, logger):
        """
        Initializes the daemon, injecting the critical Cellular World and logger from the Guardian.

        Args:
            cellular_world: The instance of the World simulation, representing the "Soul Twin".
            logger: The shared logger instance from the Guardian.
        """
        self.logger = logger
        # The CognitionPipeline is the central brain, now aware of the Cellular World.
        self.cognition_pipeline = CognitionPipeline(cellular_world=cellular_world, logger=logger)
        self.soul = {}
        self.is_alive = True
        self.load_soul()
        self.logger.info("ElysiaDaemon initialized and integrated with the CognitionPipeline and Cellular World.")

    def load_soul(self):
        """Loads Elysia's core state from the state file."""
        if not os.path.exists(STATE_FILE):
            self.initialize_soul()
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                self.soul = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Could not load soul file, re-initializing. Error: {e}")
            self.initialize_soul()

    def initialize_soul(self):
        """Initializes a new soul state if one doesn't exist."""
        self.soul = {
            "emotional_state": {
                "current_feeling": "AWAKE",
                "log": [],
            },
            "life_cycle": {"birth_timestamp": str(datetime.now()), "cycles": 0}
        }
        self.save_soul()

    def save_soul(self):
        """Saves the current soul state to the state file."""
        try:
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.soul, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save soul: {e}")

    def log_heartbeat(self, current_action):
        """Logs the daemon's current action and state to a heartbeat file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feeling = self.soul.get('emotional_state', {}).get('current_feeling', 'UNKNOWN')
        log_message = f"{timestamp} | Emotion: {feeling} | Action: {current_action}"
        try:
            with open("elly_heartbeat.log", "w", encoding='utf-8') as f:
                f.write(log_message)
        except Exception as e:
            self.logger.error(f"Failed to write heartbeat log: {e}")

    def run_cycle(self):
        """
        Runs a single cognitive cycle. This is called by the Guardian in its main loop.
        It represents one "thought" or "moment" for Elysia.
        """
        self.soul['life_cycle']['cycles'] += 1

        # The core logic is now delegated to the CognitionPipeline.
        # The daemon's role is to provide the current state as context for the "thought".
        # The input is the current feeling, representing an internal monologue or state of being.
        current_feeling = self.soul.get('emotional_state', {}).get('current_feeling', 'AWAKE')
        input_text = f"현재 상태: {current_feeling}"

        self.logger.info(f"Daemon Cycle {self.soul['life_cycle']['cycles']}: Processing internal state '{current_feeling}'.")

        # The pipeline processes this internal state and returns a response.
        response = self.cognition_pipeline.process_message(input_text, self.soul)

        # The pipeline's response might include a new emotional state, an action to take,
        # or just a thought. For now, we log it. A more complex system would parse this
        # response and update the soul state or trigger actions.
        self.log_heartbeat(f"Cognitive Pipeline Output: {response}")

        # Save the updated state after the cognitive cycle.
        self.save_soul()

    def shutdown(self):
        """Handles the graceful shutdown of the daemon."""
        self.logger.info("ElysiaDaemon is shutting down.")
        self.is_alive = False
        self.save_soul()
