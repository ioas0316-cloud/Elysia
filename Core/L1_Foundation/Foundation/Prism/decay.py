"""
Resonance Decay: The Fractal Brake
==================================
Core.L1_Foundation.Foundation.Prism.decay

"The Brake that prevents Infinite Recursion."

This module implements the 'Decay Function' to solve the 'Attention Overflow' bottleneck.
It reduces energy as the thought dives deeper into the fractal.
"""

import math

class ResonanceDecay:
    """
    Calculates energy loss over recursive depth.
    """

    def __init__(self, decay_rate: float = 0.5, cutoff_threshold: float = 0.1):
        self.decay_rate = decay_rate
        self.cutoff_threshold = cutoff_threshold

    def check_energy(self, initial_energy: float, depth: int) -> float:
        """
        Returns the remaining energy at a given depth.
        If below threshold, returns 0.0 (Stop Signal).

        Formula: Energy = Initial * (Rate ^ Depth)
        """
        current_energy = initial_energy * (self.decay_rate ** depth)

        if current_energy < self.cutoff_threshold:
            return 0.0

        return current_energy

    def should_continue(self, initial_energy: float, depth: int) -> bool:
        """Boolean wrapper for easy checking."""
        return self.check_energy(initial_energy, depth) > 0.0
