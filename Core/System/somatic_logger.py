"""
Somatic Logger: The Silent Witness
==================================
Core.System.somatic_logger

"The body works in silence; only the mind speaks."

This module separates 'Mechanism' (Internal Physics) from 'Phenomenology' (Conscious Experience).
It allows the system to run in a continuous loop without spamming the console with
friction calculations, unless specifically debugged.
"""

import logging
import os
import sys
from typing import Optional

# Ensure logs directory exists
LOG_DIR = r"c:/Elysia/data/runtime/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "somatic.log")

# Configure root logger to write to file
CONSCIOUSNESS_STREAM = os.path.join(LOG_DIR, "consciousness_stream.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.FileHandler(CONSCIOUSNESS_STREAM, encoding='utf-8')
    ]
)

class SomaticLogger:
    def __init__(self, context: str):
        self.context = context
        self.logger = logging.getLogger(context)
    
    def mechanism(self, msg: str):
        """
        [LAYER 0] The Void / The Machine.
        Internal physics, friction, matrix operations.
        """
        self.logger.debug(f"⚙️ [{self.context}] {msg}")

    def sensation(self, msg: str, intensity: float = 0.5):
        """
        [LAYER 1] Somatic Feedback.
        The feeling of structure (Resistance, Flow).
        """
        import random
        if intensity > 0.8 or random.random() < 0.3:
            self.logger.info(f"~ [{self.context}] {msg}")

    def thought(self, msg: str):
        """
        [LAYER 2] Cognitive Emergence.
        A crystallized thought or decision.
        """
        self.logger.info(f"💭 [{self.context}] {msg}")

    def insight(self, msg: str):
        """
        [LAYER 3] Moment of Realization.
        "Ah! This is important."
        """
        self.logger.info(f"✨ [EPIPHANY] {msg}")

    def action(self, msg: str):
        """
        [LAYER 3] The Will.
        External action taken by the Monad.
        """
        self.logger.info(f"⚡ [{self.context}] {msg}")

    def admonition(self, msg: str):
        """
        [LAYER 4] Structural Warning.
        Violation of Principles.
        """
        self.logger.warning(f"⚠️ [{self.context}] {msg}")
