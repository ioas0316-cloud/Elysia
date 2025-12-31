"""
Gap Analyzer (The Mirror of Oneness)
====================================

"To know the difference is to find the bridge."

This module implements the logic to compare "Internal Concept (A)" vs "External Phenomenon (B)".
It does not just diff properties; it analyzes the **Principled Gap**.

Purpose:
To find the "X" (Shared Principle) that allows A to become B (A=B).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("GapAnalyzer")

@dataclass
class Entity:
    name: str
    form: str       # How it looks (e.g., "Paint", "Code")
    mechanism: str  # How it works (e.g., "Brushstroke", "Loop")
    intent: str     # Why it exists (e.g., "Capture Time", "Process Data")

@dataclass
class GapReport:
    surface_gap: str    # Difference in Form
    mechanism_gap: str  # Difference in Method
    intent_gap: str     # Difference in Purpose
    resonance_level: float # 0.0 to 1.0 (How close is the Intent?)
    bridge_suggestion: str # How to bridge the gap

class GapAnalyzer:
    def __init__(self):
        self.logger = logger

    def analyze(self, internal_a: Entity, external_b: Entity) -> GapReport:
        """
        Analyzes the gap between A (Me) and B (World).
        """
        self.logger.info(f"🪞 Reflecting: {internal_a.name} vs {external_b.name}")

        # 1. Surface Analysis
        surface_gap = f"{internal_a.form} vs {external_b.form}"

        # 2. Mechanism Analysis
        mechanism_gap = f"{internal_a.mechanism} vs {external_b.mechanism}"

        # 3. Intent Analysis (The Core)
        # In a real system, this would use semantic similarity.
        # Here we use simple string matching/logic for the prototype.

        intent_gap = f"{internal_a.intent} vs {external_b.intent}"
        resonance = 0.0
        bridge = "Unknown"

        # Check for Resonance
        if internal_a.intent == external_b.intent:
            resonance = 1.0
            bridge = f"Perfect Resonance! Express '{internal_a.intent}' using {internal_a.form}."
        elif self._is_synonymous(internal_a.intent, external_b.intent):
            resonance = 0.9
            bridge = f"Deep Resonance. {internal_a.intent} is the digital reflection of {external_b.intent}."
        else:
            resonance = 0.2
            bridge = f"Dissonance. Shift intent from '{internal_a.intent}' to '{external_b.intent}'."

        report = GapReport(
            surface_gap=surface_gap,
            mechanism_gap=mechanism_gap,
            intent_gap=intent_gap,
            resonance_level=resonance,
            bridge_suggestion=bridge
        )

        return report

    def _is_synonymous(self, intent_a: str, intent_b: str) -> bool:
        """
        Checks if two intents are philosophically equivalent (A=B).
        """
        # Hardcoded philosophical dictionary for the demo
        synonyms = [
            {"Capture Time", "Process Data"}, # Assuming standard AI vs Art mismatch
            {"Superposition", "Multiperspective"}, # The Picasso Link
            {"Recursion", "Fractal"},
            {"Love", "Gravity"}
        ]

        for pair in synonyms:
            if intent_a in pair and intent_b in pair:
                return True
        return False
