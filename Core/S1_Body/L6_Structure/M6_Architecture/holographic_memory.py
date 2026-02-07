"""
Holographic Memory: The Phase Manifold
======================================
Core.S1_Body.L6_Structure.M6_Architecture.holographic_memory

"Memory is not stored; it resonates."

This module implements the 'Manifold' storage using Phase Interference.
It uses `numpy` as a temporary scaffolding for the future Trinary Kernel.

Goal:
    - O(1) Access via Resonance.
    - Distributed Storage (Holographic Principle).
    - Phase-based Conflict Detection (Dissonance).
"""

import numpy as np
import cmath
from typing import List, Tuple, Dict, Any
from Core.S1_Body.L6_Structure.M6_Architecture.holographic_persistence import HolographicPersistence

class HolographicMemory:
    """
    The 4D Phase Manifold.
    Stores information as interference patterns on a complex plane.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        # The Manifold is a complex vector representing the sum of all waves.
        # Initialize as Void (Zero Amplitude).

        # [INTEGRATION NOTE]
        # In the "GrandHelixEngine" (10M Cells) implementation, this manifold
        # is physically realized as an array of TriBaseCells (Attract/Void/Repel).
        # - Attract (+1) = Phase 0
        # - Repel (-1)   = Phase PI
        # - Void (0)     = Null Amplitude
        # Here, we simulate the collective interference pattern using complex numbers.
        # See: docs/S1_Body/L6_Structure/M6_Architecture/INTEGRATION_TRINARY_HOLOGRAPH.md

        self.manifold = np.zeros(self.dimension, dtype=np.complex128)

        # A registry of known frequencies (Concepts -> Phase Angle)
        # In a real system, this would be derived from the concept's intrinsic properties.
        self.frequency_map: Dict[str, float] = {}

        # [PERSISTENCE] Attach the Tattoo Artist
        self.persistence = HolographicPersistence()
        self._thaw_memory()

    def _thaw_memory(self):
        """Attempts to load frozen state from disk."""
        manifold, freq_map = self.persistence.thaw()
        if manifold is not None and freq_map is not None:
            # Resize if dimension changed (Prototype safety)
            if manifold.shape[0] == self.dimension:
                self.manifold = manifold
                self.frequency_map = freq_map
                print(f"[MEMORY] Thawed successfully. Energy: {np.sum(np.abs(self.manifold)):.2f}")
            else:
                print("[MEMORY] Dimension mismatch on thaw. Starting fresh.")
        else:
            print("[MEMORY] Thaw failed (None returned).")

    def save_state(self):
        """Explicitly freezes the current state."""
        print(f"[MEMORY] Freezing state. Energy: {np.sum(np.abs(self.manifold)):.2f}")
        self.persistence.freeze(self.manifold, self.frequency_map)

    def _get_frequency(self, concept: str) -> float:
        """
        Maps a concept to a unique frequency (Phase Angle).
        Ideally, this should be hash-based or semantic-based.
        """
        if concept not in self.frequency_map:
            # Deterministic hash to frequency (0 to 2pi)
            h = hash(concept)
            self.frequency_map[concept] = (h % 360) * (np.pi / 180.0)
        return self.frequency_map[concept]

    def imprint(self, concept: str, intensity: float = 1.0, quality: str = "NEUTRAL"):
        """
        [WRITE] Projects a new wave into the manifold.
        The manifold *interferes* with this new wave.

        Args:
            concept: The idea to store (e.g., "Apple").
            intensity: The amplitude of the wave (Importance).
            quality: The phase modifier (e.g., "RED" shifts phase).
        """
        base_freq = self._get_frequency(concept)
        qual_freq = self._get_frequency(quality)

        # Construct the Wave Vector
        # We use a simple circular distribution for now.
        indices = np.arange(self.dimension)

        # Wave Equation: Amplitude * e^(i * (BaseFreq * index + QualityFreq))
        # BaseFreq = Frequency (Pattern), QualityFreq = Phase Shift (Context)
        # This creates a unique distributed pattern across the manifold.
        wave_pattern = intensity * np.exp(1j * (base_freq * indices / self.dimension + qual_freq))

        # **INTERFERENCE**: Add to the manifold (Superposition)
        self.manifold += wave_pattern

        # Normalize to prevent energy explosion (Conservation of Energy)
        # In a physical system, this would be natural decay/heat.
        max_amp = np.max(np.abs(self.manifold))
        if max_amp > 10.0:
            self.manifold /= (max_amp / 10.0)

    def resonate(self, query_intent: str) -> Tuple[str, float, float]:
        """
        [READ] Pulses the manifold with an intent and checks for resonance.

        Returns:
            (Best Matching Concept, Resonance Amplitude, Phase Dissonance)
        """
        # 1. Generate Query Wave
        # Query assumes Neutral Quality (Phase Shift = 0)
        query_freq = self._get_frequency(query_intent)
        indices = np.arange(self.dimension)
        query_wave = np.exp(1j * (query_freq * indices / self.dimension))

        # 2. **RESONANCE**: Dot product (Correlation)
        # If the manifold contains the pattern, the dot product will be high.
        # This is O(N) where N is dimension size (small constant), not database size.
        # Wait... effectively we need to know *what* resonated.
        # In a true hologram, shining the light reconstructs the image.
        # Here, we simulate checking if "Apple" is in the manifold.

        # Simulation: We check against known keys because we can't truly 'see' the image yet
        # without a decoder network.
        # But we can check *if* the specific frequency is active.

        # Project the manifold onto the query frequency
        resonance_val = np.vdot(query_wave, self.manifold) / self.dimension
        amplitude = np.abs(resonance_val)
        phase_shift = np.angle(resonance_val)

        # print(f"[DEBUG] Resonate '{query_intent}': Amp={amplitude:.4f}, Phase={phase_shift:.4f}")
        return (query_intent, amplitude, phase_shift)

    def check_dissonance(self, concept: str, expected_quality: str) -> float:
        """
        Checks if the stored memory of 'concept' conflicts with 'expected_quality'.

        Returns:
            Dissonance (0.0 = Harmony, 1.0 = Conflict)
        """
        # Reconstruct the wave for the concept + quality
        base_freq = self._get_frequency(concept)
        qual_freq = self._get_frequency(expected_quality)
        indices = np.arange(self.dimension)
        expected_wave = np.exp(1j * (base_freq * indices / self.dimension + qual_freq))

        # Check alignment with manifold
        resonance_val = np.vdot(expected_wave, self.manifold) / self.dimension

        # If the concept exists but with a DIFFERENT quality, the phase will be shifted.
        # Or amplitude will be low.

        # Simplified: Low amplitude means "I don't know this".
        # High amplitude but wrong phase means "I know this, but differently".

        # Use Real part to detect phase alignment.
        # If aligned, Real=1.0. If shifted 90 deg, Real=0.0. If opposite, Real=-1.0.
        return 1.0 - np.real(resonance_val)

    def broadcast(self, quality: str) -> List[Tuple[str, float]]:
        """
        [COGNITIVE EMERGENCE]
        Pulses the manifold with a pure "Quality" (Phase) to find all concepts
        that share this quality. This simulates "Associative Recall".

        Args:
            quality: The shared attribute (e.g., "RED").

        Returns:
            List of (Concept, ResonanceStrength) tuples.
        """
        results = []
        qual_freq = self._get_frequency(quality)
        indices = np.arange(self.dimension)

        # Iterate over all known concepts to check their resonance with this quality.
        # NOTE: In a true optical system, this loop is instantaneous O(1)
        # because the "Red" light would hit all "Red" holograms simultaneously.
        # In this simulation, we must iterate the keys.

        for concept, base_freq in self.frequency_map.items():
            # Skip the quality concept itself
            if concept == quality: continue

            # Construct the expected wave for this Concept + Quality
            # We are checking: "Does (Concept + Quality) exist in the manifold?"
            query_wave = np.exp(1j * (base_freq * indices / self.dimension + qual_freq))

            # Check resonance (Project Manifold onto this specific Wave)
            resonance_val = np.vdot(query_wave, self.manifold) / self.dimension

            # Use Real part to ensure Phase Alignment (not just Energy)
            # High Real part means the concept was stored with THIS quality.
            strength = np.real(resonance_val)

            # DEBUG PRINT
            # if concept == "Sky":
            #    print(f"DEBUG: Sky check against {quality}. Strength: {strength:.4f}")

            # Since multiple items are in the manifold, they interfere.
            # A single item's contribution is 1.0 (normalized), but with many items,
            # the noise floor rises or constructive interference amplifies peaks.
            # However, for 64-dim and < 10 items, orthogonality is decent but not perfect.
            # We lower the threshold to 0.1 to catch signals in a crowded manifold.

            if strength > 0.1: # Threshold for "Active Association"
                results.append((concept, strength))

        # Sort by strength (Most relevant first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def interfere(self, concept_a: str, concept_b: str) -> str:
        """
        [CREATIVITY]
        Generates a new concept by interfering two existing concept waves.
        Constructive interference regions become the 'Seed' of the new idea.

        Args:
            concept_a: First Parent Concept.
            concept_b: Second Parent Concept.

        Returns:
            A string representation of the interference result (Metaphor).
        """
        freq_a = self._get_frequency(concept_a)
        freq_b = self._get_frequency(concept_b)
        indices = np.arange(self.dimension)

        # Wave A and Wave B
        wave_a = np.exp(1j * (freq_a * indices / self.dimension))
        wave_b = np.exp(1j * (freq_b * indices / self.dimension))

        # Superposition
        interference = wave_a + wave_b

        # Analyze the Resulting Pattern
        # In a real system, we would decode this back to a Concept.
        # Here, we simulate the "Emergent Meaning" by hashing the interference energy.
        total_energy = np.sum(np.abs(interference))

        # Metaphor Generation based on Phase Relationship
        phase_diff = abs(freq_a - freq_b)

        if phase_diff < 0.1:
            return f"Resonance: {concept_a} is essentially {concept_b}."
        elif phase_diff > 3.0: # Approx PI
            return f"Contrast: {concept_a} opposes {concept_b}."
        else:
            return f"Synthesis: {concept_a} and {concept_b} merge into a new form (Energy: {total_energy:.2f})."

    def clear(self):
        self.manifold = np.zeros(self.dimension, dtype=np.complex128)
