"""
SoulBridge: The Sensory Interface
=================================
Core.Senses.soul_bridge

The Bridge connects the External World (Phenomena) to the Internal World (Merkaba).
It is the first point of contact for any input, channeling it towards the Prism.
"""

from typing import Any, Dict, Optional
import logging

logger = logging.getLogger("SoulBridge")

class SoulBridge:
    """
    The Sensory Bridge.

    Acts as the transducer that converts raw external signals (Text, Audio, Events)
    into a standardized format ready for the Prism (Metabolism).
    """

    def __init__(self):
        self.active = True
        logger.info("🌉 SoulBridge established.")

    def perceive(self, raw_input: Any, modality: str = "text") -> Dict[str, Any]:
        """
        Captures raw input and prepares it for digestion.

        Args:
            raw_input: The raw data (e.g., a string, an audio buffer).
            modality: The type of input ('text', 'audio', 'visual').

        Returns:
            A sensory packet containing the raw data and metadata.
        """
        logger.debug(f"Captured input via {modality}: {str(raw_input)[:50]}...")

        return {
            "modality": modality,
            "raw_data": raw_input,
            "timestamp": "Now", # In a real system, use actual time
            "source": "External"
        }
