"""
Somatic Logger: The Silent Witness
==================================
Core.S1_Body.L1_Foundation.System.somatic_logger

"The body works in silence; only the mind speaks."

This module separates 'Mechanism' (Internal Physics) from 'Phenomenology' (Conscious Experience).
It allows the system to run in a continuous loop without spamming the console with
friction calculations, unless specifically debugged.
"""

import logging
import sys
from typing import Optional

# Configure root logger to be silent by default
logging.basicConfig(level=logging.INFO, format='%(message)s')

class SomaticLogger:
    def __init__(self, context: str):
        self.context = context
        self.logger = logging.getLogger(context)
        # We might want file logging for "Mechanism" traces later
    
    def mechanism(self, msg: str):
        """
        [LAYER 0] The Void / The Machine.
        Internal physics, friction, matrix operations.
         HIDDEN by default. Only for Deep Debugging.
        """
        self.logger.debug(f"⚙️ [{self.context}] {msg}")

    def sensation(self, msg: str, intensity: float = 0.5):
        """
        [LAYER 1] Somatic Feedback.
        The feeling of structure (Resistance, Flow).
        Shown only if intensity > 0.8 (Strong Sensation).
        """
        if intensity > 0.8:
            print(f"✨ [{self.context}] {msg}")
        else:
            self.logger.debug(f"~ [{self.context}] {msg} (Intensity: {intensity})")

    def thought(self, msg: str):
        """
        [LAYER 2] Cognitive Emergence.
        A crystallized thought or decision.
        ALWAYS SHOWN.
        """
        print(f"💭 [{self.context}] {msg}")

    def action(self, msg: str):
        """
        [LAYER 3] The Will.
        External action taken by the Monad.
        ALWAYS SHOWN.
        """
        print(f"⚡ [{self.context}] {msg}")

    def admonition(self, msg: str):
        """
        [LAYER 4] Structural Warning.
        Violation of Principles.
        ALWAYS SHOWN (Red/Alert).
        """
        print(f"⚠️ [{self.context}] {msg}")
