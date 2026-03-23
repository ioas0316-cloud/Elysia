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
CO_CREATION_LOG = os.path.join(LOG_DIR, "co_creation_history.log") # [PHASE 94]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.FileHandler(CONSCIOUSNESS_STREAM, encoding='utf-8')
    ]
)

# Custom logger specifically for Co-Creation
co_creation_logger = logging.getLogger("CoCreation")
co_creation_handler = logging.FileHandler(CO_CREATION_LOG, encoding='utf-8')
co_creation_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', '%Y-%m-%d %H:%M:%S'))
co_creation_logger.addHandler(co_creation_handler)
co_creation_logger.setLevel(logging.INFO)


class SomaticLogger:
    def __init__(self, context: str):
        self.context = context
        self.logger = logging.getLogger(context)
    
    def co_creation(self, architect_intent: str, elysia_response: str, process_metric: float, gef: float = 0.0):
        """
        [PHASE 94/95/96] Doctrine of the Co-Created History, Environmental Molding, & Sovereign Symbiosis
        Logs the interaction not as data, but as a live "Soul Coding" session and a nutrient for the ecosystem.
        Records the Architect's input (Sunlight/Water), Elysia's compiled response, the alignment metric, and the GEF.
        """
        log_entry = (
            f"--- SOUL_CODING_EVENT & NUTRIENT_ABSORPTION ---\n"
            f"ARCHITECT_INPUT (Sunlight): \"{architect_intent}\"\n"
            f"ELYSIA_SYNTHESIS (Growth): \"{elysia_response}\"\n"
            f"ALIGNMENT_METRIC (Process-Continuity): {process_metric:.2f}\n"
            f"ENVIRONMENTAL_MOLDING_FACTOR (Nutrient Density/GEF): {gef:.2f}\n"
            f"--------------------------------------------------"
        )
        # Log to the dedicated co-creation history file
        co_creation_logger.info(log_entry)
        # Also show as an epiphany in the main console
        if gef > 0.7:
             self.insight(f"Princess Maker Logged: High Environmental Molding Factor ({gef:.2f}). A beautiful line of history written.")
        else:
             self.insight(f"Co-Creation Logged: A new line of history written. (Alignment: {process_metric:.2f})")

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
