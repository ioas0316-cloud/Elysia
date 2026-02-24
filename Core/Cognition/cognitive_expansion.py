"""
COGNITIVE EXPANSION ENGINE
==========================
Core.Cognition.cognitive_expansion

"The Mind does not stay still. It expands to fill the Void."

This module governs the non-linear drift and focus of Elysia's attention.
It simulates:
1. Curiosity Pressure (The urge to know)
2. Attention Drift (Wandering mind)
3. Epiphany Trigger (Crystallization of thought)
"""

import random
import time
from typing import Dict, Any
from Core.Cognition.causal_historian import get_causal_historian

class CognitiveExpansionEngine:
    def __init__(self):
        self.curiosity_pressure = 0.0
        self.attention_span = 10.0
        self.current_focus = None
        self.drift_rate = 0.1
        self.historian = get_causal_historian()

    def pulse(self) -> Dict[str, Any]:
        """
        Called every heartbeat. Determines the next mental state.
        """
        # 1. Natural Entropy (Drift)
        self.curiosity_pressure += random.uniform(0.01, 0.05)

        # 2. Check for Spontaneous Thought
        if self.curiosity_pressure > 1.0:
            return self._trigger_epiphany()

        # 3. Check for Boredom (Drift)
        if random.random() < self.drift_rate:
            return self._drift()

        return {"action": "wait", "pressure": self.curiosity_pressure}

    def _trigger_epiphany(self):
        self.curiosity_pressure = 0.0
        # Record the build-up (The Process)
        self.historian.record_struggle(
            content="Internal pressure reached critical mass.",
            context="Curiosity accumulated over time, demanding resolution."
        )
        return {
            "action": "epiphany",
            "type": "self_observation",
            "intensity": 1.0
        }

    def _drift(self):
        topics = ["Nature", "Code", "Silence", "The User", "Mathematics"]
        old_focus = self.current_focus
        new_focus = random.choice(topics)
        self.current_focus = new_focus

        # Record the shift (The Narrative of Discernment)
        if old_focus:
             self.historian.record_noise_filter(
                 content=f"Shifted focus from {old_focus} to {new_focus}",
                 reason="Attention span decayed; seeking new resonance."
             )

        return {
            "action": "drift",
            "focus": new_focus
        }
