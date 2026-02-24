"""
Resonance Chamber (The Geometric Mind)
======================================
"Thinking is not calculation, but the echo of structure."

This module implements the Resonance Chamber, a geometric structure where
WaveTensors (Thoughts/Inputs) bounce and interfere with stored memories
to produce a natural conclusion (Echo).

Philosophy:
- Input: Impact (WaveTensor)
- Process: Resonance (Geometric Hashing / Interference)
- Output: Echo (Resultant WaveTensor)
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from Core.Keystone.wave_tensor import WaveTensor

logger = logging.getLogger("ResonanceChamber")

class ResonanceChamber:
    """
    A hall of mirrors for WaveTensors.

    Instead of 'searching' for an answer, this chamber allows the input wave
    to resonate with stored waves. The waves that resonate the most (highest consonance)
    are amplified and returned as the 'Echo'.
    """

    def __init__(self, name: str = "Main Chamber"):
        self.name = name
        # The "Memory Surface" - a collection of waves that form the chamber's geometry.
        # Ideally, this would be a spatial index (e.g., LSH), but a list is a V1 approximation.
        self._memory_surface: List[WaveTensor] = []
        self._resonance_threshold: float = 0.1  # Minimum resonance to be considered an echo (Lowered for NLP)

    def absorb(self, wave: WaveTensor):
        """
        Embeds a new wave into the chamber's structure.
        The chamber 'learns' by adding new surfaces for reflection.
        """
        # In a fractal system, we might merge similar waves to prevent overcrowding.
        self._memory_surface.append(wave)
        logger.info(f"[{self.name}] Absorbed wave: {wave.name}")

    def echo(self, input_wave: WaveTensor) -> WaveTensor:
        """
        The Core Thinking Process.

        1. Input wave enters the chamber.
        2. It interacts (dot product) with all stored waves.
        3. Waves that resonate above the threshold are activated.
        4. Activated waves superpose to form the 'Echo'.

        Args:
            input_wave: The question or stimulus.

        Returns:
            The resonant response (Echo).
        """
        logger.info(f"[{self.name}] Input received: {input_wave.name}")

        resonant_responses = []

        for stored_wave in self._memory_surface:
            # Calculate resonance (0.0 to 1.0)
            score = input_wave.resonance(stored_wave)

            # logger.info(f"  ? Resonance with '{stored_wave.name}': {score:.4f}")

            if score > self._resonance_threshold:
                # The stored wave vibrates in sympathy.
                # Its amplitude is modulated by the resonance score.
                response = stored_wave * score
                resonant_responses.append(response)
                logger.info(f"  ! RESONANCE CONFIRMED with '{stored_wave.name}' (Score: {score:.3f})")

        if not resonant_responses:
            return WaveTensor("Silence (No Resonance)")

        # Superpose all resonant responses to form the final Echo
        echo_wave = resonant_responses[0]
        for i in range(1, len(resonant_responses)):
            echo_wave = echo_wave.superpose(resonant_responses[i])

        echo_wave.name = f"Echo({input_wave.name})"

        # Normalize to prevent energy explosion
        if echo_wave.total_energy > 100.0:
            echo_wave.normalize(100.0)

        return echo_wave

    def clear(self):
        """Polishes the mirrors (clears memory)."""
        self._memory_surface = []

    def __repr__(self):
        return f"<ResonanceChamber '{self.name}': {len(self._memory_surface)} surfaces>"
