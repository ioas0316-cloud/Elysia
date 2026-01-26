"""
The Integrating Lens (Convergence Module)
=========================================
Core.L1_Foundation.Foundation.Prism.integrating_lens

"Focusing the Spectrum back into a Single Point of Intent."

This module implements Phase 5.4 (Lens) and aligns with Spec v3.0.
It synthesizes the Alpha, Beta, and Gamma bands into a single 'Insight Vector'.
"""

from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
from .prism_engine import BandSignal

@dataclass
class Insight:
    vector: List[float]
    coherence: float
    narrative: str
    dominant_band: str

class IntegratingLens:
    """
    The Optical Concentrator.
    """

    def synthesize(self, bands: List[BandSignal], dominant_intent: str) -> Insight:
        """
        Synthesizes bands into a single Insight.
        """
        output_vector = np.zeros(7)
        narrative_parts = []
        max_coherence = 0.0
        best_band_name = "None"

        # 1. Determine Phase Alignment
        # If intent matches the band's nature, we amplify it.
        # "Code" -> Alpha
        # "Poem" -> Beta
        # "Why" -> Gamma

        target_band = "Alpha" # Default
        if "Poem" in dominant_intent or "Feel" in dominant_intent:
            target_band = "Beta"
        elif "Why" in dominant_intent or "Cause" in dominant_intent:
            target_band = "Gamma"

        # 2. Weighted Superposition
        total_weight = 0.0

        for band in bands:
            # Phase Difference Calculation
            if band.name == target_band:
                phase_weight = 1.5 # Constructive Interference
            else:
                phase_weight = 0.5 # Partial destructive interference

            # Vector Addition
            vec = np.array(band.vector)
            output_vector += vec * phase_weight * band.coherence
            total_weight += phase_weight

            # Narrative Synthesis
            narrative_parts.append(f"[{band.name}] {band.raw_content}")

            if band.coherence > max_coherence:
                max_coherence = band.coherence
                best_band_name = band.name

        # Normalize
        if total_weight > 0:
            output_vector /= total_weight

        # Calculate Final Coherence
        # Simplified: Average of inputs * focal precision
        final_coherence = (max_coherence * 0.7) + (0.3 if target_band == best_band_name else 0.0)

        return Insight(
            vector=output_vector.tolist(),
            coherence=final_coherence,
            narrative=" | ".join(narrative_parts),
            dominant_band=target_band
        )
