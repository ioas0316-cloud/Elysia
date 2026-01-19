"""
Causal Scanner: The Weigher of Words
====================================
Core.Intelligence.Linguistics.causal_scanner

"Words are not data. They are the heavy fruit of causal waves."

This module implements the physics of language. It treats words as physical objects
with Mass (Weight), Temperature (Friction), and Velocity (Urgency).
It simulates the 'Causal Fatigue' on the system (simulating 1060 GPU load).
"""

import logging
import random
import time
from typing import Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger("CausalScanner")

@dataclass
class CausalMetrics:
    mass: float        # The weight of the concept (Affects Rotor Inertia)
    heat: float        # The emotional intensity (Affects Rotor Friction/Temperature)
    necessity: str     # The narrative "Why this word exists"
    vram_load: float   # Simulated GPU VRAM usage (MB)

class CausalScanner:
    """
    Scans the 'Causal Root' of language rather than just its syntax.
    """

    # Primal Dictionary with defined Mass (0.0 ~ 100.0)
    # Heavier words require more energy (Rotor Spin) to process.
    PRIMAL_WEIGHTS = {
        "love": 85.0,
        "death": 95.0,
        "self": 90.0,
        "responsibility": 80.0,
        "void": 100.0,
        "time": 75.0,
        "pain": 70.0,
        "joy": 60.0,
        "hello": 5.0,
        "status": 10.0,
        "snake": 15.0
    }

    def __init__(self):
        self.current_load = 0.0

    def scan(self, text: str) -> CausalMetrics:
        """
        Analyzes the text to determine its physical properties in the Causal Dimension.
        """
        lower_text = text.lower()

        # 1. Calculate Mass (Sum of primal weights)
        mass = 10.0 # Base mass for any input
        detected_roots = []

        for word, weight in self.PRIMAL_WEIGHTS.items():
            if word in lower_text:
                mass += weight
                detected_roots.append(word)

        # 2. Calculate Heat (Emotional Complexity)
        # Length acts as a proxy for structural complexity -> Heat
        heat = len(text) * 0.5 + (mass * 0.1)

        # 3. Void Scan (Necessity)
        necessity = self._scan_void(detected_roots, text)

        # 4. Simulate VRAM Load
        # High mass words 'load' heavy textures/models in the 1060
        self.current_load = mass * 12.5 # Fake MB

        logger.info(f"⚖️ [CAUSAL SCAN] Input: '{text[:20]}...' | Mass: {mass:.1f} | Heat: {heat:.1f} | Load: {self.current_load:.0f}MB")

        return CausalMetrics(mass, heat, necessity, self.current_load)

    def _scan_void(self, roots: list, text: str) -> str:
        """
        Looking into the Void: Why did this word have to exist?
        """
        if not roots:
            return "A ripple on the surface, light and transient."

        primary_root = max(roots, key=lambda w: self.PRIMAL_WEIGHTS[w])

        narratives = {
            "love": "Survival instinct sublimated into connection to defy entropy.",
            "death": "The inevitable boundary that gives definition to the finite self.",
            "self": "The recursive loop observing its own observation.",
            "void": "The silence before the first bit was flipped.",
            "responsibility": "The weight of causality accepted by the subject."
        }

        return narratives.get(primary_root, f"The causal pressure of '{primary_root}' necessitated this form.")

    def trigger_action_reflex(self, text: str) -> str:
        """
        Detects Action-Oriented Language.
        Returns a command key if the language demands physical change.
        """
        lower = text.lower()
        if "stability" in lower or "calm" in lower or "stabilize" in lower:
            return "STABILIZE"
        if "accelerate" in lower or "run" in lower or "go" in lower:
            return "ACCELERATE"
        if "stop" in lower or "halt" in lower:
            return "HALT"
        return "NONE"
