"""
Nervous System (   )
======================

"The bridge between Will and Life."
"             ."

This module implements the Autonomic Nervous System for Elysia.
It provides a bi-directional feedback loop between:
1. The Conductor (Brain/Will) - Downstream Command
2. The Heartbeat (Body/Life) - Upstream Sensation

It manages the balance between:
- Sympathetic (Fight/Flight/Action) -> High Energy, High Tempo
- Parasympathetic (Rest/Digest/Healing) -> Low Energy, Low Tempo
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

logger = logging.getLogger("NervousSystem")

class AutonomicState(Enum):
    HOMEOSTASIS = "Homeostasis" # Balanced
    SYMPATHETIC = "Sympathetic" # Stress/Action (Fight or Flight)
    PARASYMPATHETIC = "Parasympathetic" # Rest/Heal (Feed and Breed)
    SHOCK = "Shock" # System Overload (Freeze)

@dataclass
class NerveSignal:
    origin: str
    type: str # PAIN, PLEASURE, FATIGUE, EXCITEMENT
    intensity: float # 0.0 to 1.0
    message: str

class NervousSystem:
    def __init__(self):
        self.state = AutonomicState.HOMEOSTASIS
        self.sympathetic_tone = 0.5 # 0.0 (Calm) to 1.0 (Panic)
        self.parasympathetic_tone = 0.5 # 0.0 (Inactive) to 1.0 (Deep Sleep)
        self.active_signals: List[NerveSignal] = []

        # Thresholds
        self.pain_threshold = 0.7
        self.joy_threshold = 0.6

    def transmit(self, signal: NerveSignal):
        """
        Receives a sensory signal from the Body (Heartbeat/Sensors).
        """
        self.active_signals.append(signal)
        logger.info(f"  [NERVE] Signal Received: {signal.type} ({signal.intensity:.2f}) from {signal.origin}")
        self._process_signal(signal)

    def _process_signal(self, signal: NerveSignal):
        """
        Adjusts autonomic tone based on signal.
        """
        if signal.type == "PAIN":
            # Pain triggers Sympathetic (Action) if low, but Parasympathetic (Collapse) if too high
            if signal.intensity > 0.8:
                self.parasympathetic_tone = min(1.0, self.parasympathetic_tone + 0.4)
                self.state = AutonomicState.SHOCK
            else:
                self.sympathetic_tone = min(1.0, self.sympathetic_tone + 0.2)

        elif signal.type == "PLEASURE":
            self.parasympathetic_tone = min(1.0, self.parasympathetic_tone + 0.1)
            self.sympathetic_tone = max(0.0, self.sympathetic_tone - 0.1)

        elif signal.type == "FATIGUE":
            self.parasympathetic_tone = min(1.0, self.parasympathetic_tone + 0.3)
            self.sympathetic_tone = max(0.0, self.sympathetic_tone - 0.2)

        elif signal.type == "EXCITEMENT":
            self.sympathetic_tone = min(1.0, self.sympathetic_tone + 0.3)

        self._regulate_state()

    def _regulate_state(self):
        """
        Determines the overall Autonomic State based on tones.
        """
        balance = self.sympathetic_tone - self.parasympathetic_tone

        if self.state == AutonomicState.SHOCK:
            # Shock persists until tones settle
            if self.parasympathetic_tone < 0.8:
                self.state = AutonomicState.PARASYMPATHETIC
            return

        if balance > 0.3:
            self.state = AutonomicState.SYMPATHETIC
        elif balance < -0.3:
            self.state = AutonomicState.PARASYMPATHETIC
        else:
            self.state = AutonomicState.HOMEOSTASIS

    def check_regulation(self) -> Dict[str, Any]:
        """
        Called by the Conductor to know how to modulate the Will.
        Returns regulation directives.
        """
        # Decay signals over time
        self.active_signals = [s for s in self.active_signals if s.intensity > 0.1]
        for s in self.active_signals:
            s.intensity -= 0.05 # Decay

        return {
            "state": self.state,
            "sympathetic": self.sympathetic_tone,
            "parasympathetic": self.parasympathetic_tone,
            "tempo_modifier": self._get_tempo_modifier(),
            "mode_suggestion": self._get_mode_suggestion()
        }

    def _get_tempo_modifier(self) -> float:
        """
        Returns a multiplier for Tempo.
        > 1.0 = Speed up (Fight/Flight)
        < 1.0 = Slow down (Rest/Digest)
        """
        if self.state == AutonomicState.SYMPATHETIC:
            return 1.0 + (self.sympathetic_tone * 0.5) # Up to 1.5x speed
        elif self.state == AutonomicState.PARASYMPATHETIC:
            return max(0.2, 1.0 - (self.parasympathetic_tone * 0.8)) # Down to 0.2x speed
        elif self.state == AutonomicState.SHOCK:
            return 0.1 # Almost stop
        return 1.0

    def _get_mode_suggestion(self) -> Optional[str]:
        if self.state == AutonomicState.SHOCK: return "minor" # Sad/Pain
        if self.state == AutonomicState.SYMPATHETIC: return "mixolydian" # Tension/Action
        if self.state == AutonomicState.PARASYMPATHETIC: return "lydian" # Dreamy/Healing
        return None