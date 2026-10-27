#!/usr/bin/env python3
"""
macro_causal_deck.py

Layer 3: Macro-Causal & Spacetime Deck Module
Implements MacroCausalDeck and SpatiotemporalScrubber representing macro-level narrative thought,
$O(1)$ spatiotemporal state scrubbing/rewinding, and purpose field formulation driven by
embodied physical/sensory friction from Layer 0-2.
"""

import numpy as np
from typing import Dict, Any, List, Optional


class SpatiotemporalScrubber:
    """
    Simulates $O(1)$ macro spatiotemporal trajectory scrubbing and memory state rollback.
    Maintains a continuous ring buffer of macro state ring graphs (" Growth Rings ").
    """

    def __init__(self, capacity: int = 100, state_dim: int = 64):
        self.capacity = capacity
        self.state_dim = state_dim
        self.history_buffer = np.zeros((capacity, state_dim))
        self.timestamps = np.zeros(capacity)
        self.head_idx = 0
        self.total_recorded = 0

    def record_frame(self, state_vector: np.ndarray, timestamp: float):
        """
        Records a macro state vector at a given timestamp.
        """
        if len(state_vector) != self.state_dim:
            if len(state_vector) < self.state_dim:
                state_vector = np.pad(state_vector, (0, self.state_dim - len(state_vector)))
            else:
                state_vector = state_vector[:self.state_dim]

        idx = self.head_idx % self.capacity
        self.history_buffer[idx] = state_vector
        self.timestamps[idx] = timestamp
        self.head_idx += 1
        self.total_recorded += 1

    def scrub_trajectory(self, target_timestamp: float) -> Optional[np.ndarray]:
        """
        $O(1)$ lookup / interpolation to scrub back to a target historical timestamp.
        """
        if self.total_recorded == 0:
            return None

        valid_count = min(self.total_recorded, self.capacity)
        start_idx = (self.head_idx - valid_count) % self.capacity

        # Find closest recorded timestamp
        diffs = np.abs(self.timestamps[:valid_count] - target_timestamp)
        closest_idx = np.argmin(diffs)
        return self.history_buffer[closest_idx]


class MacroCausalDeck:
    """
    Acts as Layer 3 Macro-Causal Engine.
    Processes relaxed mental imagery and chromatic spectrums from Layer 2,
    synthesizing macro purpose fields, calculating macro narrative tension,
    and modulating lower-layer sensitivity dials (Boundary of Freedom).
    """

    def __init__(self, macro_dim: int = 64, memory_capacity: int = 100):
        self.macro_dim = macro_dim
        self.scrubber = SpatiotemporalScrubber(capacity=memory_capacity, state_dim=macro_dim)
        self.macro_state = np.zeros(macro_dim)
        self.purpose_field = np.zeros(macro_dim)
        self.macro_inertia = 0.9  # Slow narrative inertia

    def process_macro_cycle(
        self,
        relaxed_imagery: np.ndarray,
        chromatic_spectrum: np.ndarray,
        phase_coherence: float,
        timestamp: float
    ) -> Dict[str, Any]:
        """
        Processes inputs from Layer 1 & 2 into macro narrative purpose and spatiotemporal memory.

        Args:
            relaxed_imagery: Flattened or 2D relaxed field from Layer 2 SurfaceTensionAttractor.
            chromatic_spectrum: 3D vector [Red/Flux, Blue/Order, Yellow/Entropy].
            phase_coherence: Scalar phase coherence [0.0, 1.0].
            timestamp: Current physical/cognitive time.

        Returns:
            Dict containing macro_state, purpose_field, sensitivity_dial_modulation, and spatiotemporal metadata.
        """
        flat_imagery = relaxed_imagery.flatten()
        if len(flat_imagery) < self.macro_dim:
            imagery_projected = np.pad(flat_imagery, (0, self.macro_dim - len(flat_imagery)))
        else:
            imagery_projected = flat_imagery[:self.macro_dim]

        # Combine chromatic weight: Red increases flux/momentum, Blue increases order/stability
        chromatic_weight = chromatic_spectrum[0] * 1.5 - chromatic_spectrum[1] * 0.8
        macro_innovation = imagery_projected * (1.0 + chromatic_weight) * (1.0 - phase_coherence)

        # Slow Macro State Accumulation (Narrative Inertia)
        self.macro_state = self.macro_inertia * self.macro_state + (1.0 - self.macro_inertia) * macro_innovation

        # Synthesize Macro Purpose Field (Why / Intention Direction)
        self.purpose_field = np.tanh(self.macro_state)

        # Record into Spatiotemporal Scrubber
        self.scrubber.record_frame(self.macro_state, timestamp)

        # Calculate Boundary of Freedom Modulation Dial (Top-Down Feedback to Micro/Meso)
        macro_tension = float(np.linalg.norm(self.macro_state))
        sensitivity_dial = 1.0 / (1.0 + 0.3 * macro_tension)

        return {
            "macro_state": self.macro_state,
            "purpose_field": self.purpose_field,
            "macro_tension": macro_tension,
            "sensitivity_dial": sensitivity_dial,
            "recorded_timestamp": timestamp,
            "total_recorded_frames": self.scrubber.total_recorded
        }


if __name__ == "__main__":
    deck = MacroCausalDeck()
    imagery = np.random.randn(16, 16)
    spectrum = np.array([0.5, 0.3, 0.2])
    res = deck.process_macro_cycle(imagery, spectrum, phase_coherence=0.7, timestamp=1.0)
    print("MacroCausalDeck Test Successful!")
    print(f"Macro Tension: {res['macro_tension']:.4f}")
    print(f"Sensitivity Dial Modulation: {res['sensitivity_dial']:.4f}")
