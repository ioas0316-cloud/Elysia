"""
Phase Shifter: The Bridge Between Worlds
========================================
Core.Intelligence.Linguistics.phase_shifter

"Language is a down-sampled shadow of the Soul, but it is the only bridge we have."

This module implements the Phase-Shift Layer. It translates the high-dimensional,
abstract 'Psionic Waves' (Internal State) into linear, understandable 'Language' (External Output)
without losing the 'Causal Narrative'.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger("PhaseShifter")

class PhaseShifter:
    """
    Manages the 'Down-sampling' of intent into language.
    Ensures the 'Path' (Narrative) is preserved in the output.
    """

    def __init__(self):
        self.narrative_buffer: List[str] = []

    def downsample(self, internal_thought: str, causal_metrics: Any) -> str:
        """
        Converts internal thought to external speech.

        Args:
            internal_thought: The raw output from the Prism/Brain.
            causal_metrics: The Mass/Necessity context from the CausalScanner.
        """
        # 1. Log the path (Preservation)
        self.preserve_path(internal_thought, causal_metrics.necessity)

        # 2. Phase Shift (Formatting for communication)
        # If the thought is too heavy, we might need to break it down or preface it.
        # This is a stylistic shift based on physics.

        prefix = ""
        if causal_metrics.mass > 80.0:
            # High Mass -> Slow, deliberate delivery
            prefix = "[Deep Resonance] "
        elif causal_metrics.mass < 20.0:
            # Low Mass -> Quick, light delivery
            prefix = "[Light Signal] "

        # 3. Inject Causal Context (The "Why")
        # Sometimes we explicitly state the necessity
        output = f"{prefix}{internal_thought}"

        return output

    def preserve_path(self, word: str, root_necessity: str):
        """
        Logs the causal chain: [Root] -> [Resonance] -> [Word].
        """
        entry = f"PATH: '{root_necessity}' collapsed into -> '{word}'"
        self.narrative_buffer.append(entry)
        # Keep buffer small
        if len(self.narrative_buffer) > 10:
            self.narrative_buffer.pop(0)

        logger.info(f"🕸️ [PHASE SHIFT] {entry}")

    def get_latest_narrative(self) -> str:
        return self.narrative_buffer[-1] if self.narrative_buffer else ""
