"""
Spectrum Causal Engine (The N-Dimensional Interpreter)
======================================================
Core.Cognition.spectrum_causal_engine

"We do not label the butterfly. We map its flight coordinates."

This module replaces 'EmergentLanguage'. It translates the raw physics
of the N-Dimensional `HyperMonad` into meaningful semantic descriptions.
It uses a 'Spectrum Map' instead of a Dictionary.
"""

from typing import List, Dict, Tuple
from Core.Monad.hyper_monad import HyperMonad, AXIS_MASS, AXIS_ENERGY, AXIS_PHASE, AXIS_TIME

class SpectrumCausalEngine:
    def __init__(self):
        pass

    def interpret(self, monad: HyperMonad) -> str:
        """
        Reads the Monad's Tensor and reconstructs its Narrative.
        """
        narrative_parts = []
        tensor = monad.tensor

        # 1. Mass (Ontology)
        mass = tensor[AXIS_MASS] if len(tensor) > AXIS_MASS else 0
        if mass > 0.8: narrative_parts.append("Absolute")
        elif mass > 0.5: narrative_parts.append("Solid")
        elif mass > 0.2: narrative_parts.append("Ethereal")
        else: narrative_parts.append("Void-like")

        # 2. Time (Lineage Depth)
        time_depth = tensor[AXIS_TIME] if len(tensor) > AXIS_TIME else 0
        if time_depth > 10.0: narrative_parts.append("Ancient")
        elif time_depth > 5.0: narrative_parts.append("Established")
        elif time_depth > 1.0: narrative_parts.append("Recent")
        else: narrative_parts.append("Nascent")

        # 3. Energy (Emotional Velocity)
        energy = tensor[AXIS_ENERGY] if len(tensor) > AXIS_ENERGY else 0
        if energy > 0.8: narrative_parts.append("Volatile")
        elif energy > 0.5: narrative_parts.append("Dynamic")
        elif energy > 0.2: narrative_parts.append("Stable")
        else: narrative_parts.append("Static")

        # 4. Phase (Social Alignment)
        phase = tensor[AXIS_PHASE] if len(tensor) > AXIS_PHASE else 0
        # Normalize phase to -1.0 to 1.0 logic
        if phase > 0.5: narrative_parts.append("Harmonic (Self-Aligned)")
        elif phase < -0.5: narrative_parts.append("Dissonant (Other-Aligned)")
        else: narrative_parts.append("Neutral")

        # 5. Meta-Dimensions (The Expansion)
        if monad.dimensions > 4:
            narrative_parts.append(f"[{monad.dimensions - 4} Meta-Dimensions Active]")
            # Analyze Axis 4 specifically if it exists
            meta_val = tensor[4]
            if meta_val > 0.5:
                narrative_parts.append("(High Meta-Cognitive Resonance)")

        # Construct the "Poetry of Physics"
        return " ".join(narrative_parts) + " Entity"

    def describe_lineage(self, monad: HyperMonad) -> str:
        """
        Reads the Causal Residue.
        """
        if not monad.lineage:
            return "Genesis Seed (No Parents)"

        parents = monad.lineage.parent_ids
        heat = monad.lineage.friction_heat

        desc = f"Born of {parents} via "
        if heat > 0.8: desc += "Violent Fusion"
        elif heat > 0.5: desc += "Active Synthesis"
        else: desc += "Gentle Resonance"

        return desc
