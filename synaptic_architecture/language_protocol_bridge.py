"""
Language Protocol Bridge Module.

Implements 1:1 isomorphic grounding between self-emergent internal macro axioms/potential
geometries and external language/symbolic protocols without external statistical guessing.
Includes inter-subjective coordination for multi-agent mirror resonance and protocol alignment.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class LanguageProtocolBridge:
    """
    Bridges internal macro potential invariants and external language/symbols via 1:1 topological isomorphism.
    """

    def __init__(self) -> None:
        """Initialize symbol protocol vocabulary mappings."""
        # Vocabulary mapping internal physical properties to external symbolic terms
        self.symbol_groundings: Dict[str, Dict[str, Any]] = {
            "Constraint": {"min_pot": 0.4, "max_pot": 1.0, "min_coherence": 0.6},
            "Entropy": {"max_coherence": 0.2, "min_friction": 0.5},
            "Resonance": {"min_coherence": 0.7, "max_friction": 0.2},
            "Boundary": {"min_gradient": 0.3},
            "Flow": {"min_valence": 0.3},
        }

    def align_internal_to_external_symbol(
        self,
        macro_potential_mean: float,
        coherence_mean: float,
        friction_mean: float,
        valence: float,
    ) -> Dict[str, Any]:
        """
        Map internal physical dynamics directly to external symbolic terms based on 1:1 isomorphism.

        Returns:
            Dictionary containing best matched ground symbol and isomorphism score.
        """
        best_symbol = "Unknown"
        best_score = -1.0

        for symbol, properties in self.symbol_groundings.items():
            score = 0.0
            checks = 0

            if "min_pot" in properties:
                score += 1.0 if macro_potential_mean >= properties["min_pot"] else 0.0
                checks += 1
            if "min_coherence" in properties:
                score += 1.0 if coherence_mean >= properties["min_coherence"] else 0.0
                checks += 1
            if "max_coherence" in properties:
                score += 1.0 if coherence_mean <= properties["max_coherence"] else 0.0
                checks += 1
            if "max_friction" in properties:
                score += 1.0 if friction_mean <= properties["max_friction"] else 0.0
                checks += 1
            if "min_friction" in properties:
                score += 1.0 if friction_mean >= properties["min_friction"] else 0.0
                checks += 1
            if "min_valence" in properties:
                score += 1.0 if valence >= properties["min_valence"] else 0.0
                checks += 1

            norm_score = score / max(1, checks)
            if norm_score > best_score:
                best_score = norm_score
                best_symbol = symbol

        return {
            "grounded_symbol": best_symbol,
            "isomorphism_score": best_score,
            "phase_aligned": best_score >= 0.75,
        }

    def inter_subjective_mirror_resonance(
        self,
        self_potential: np.ndarray,
        other_potential: np.ndarray,
    ) -> Dict[str, float]:
        """
        Pillar 4: Inter-subjective coordination.
        Computes phase alignment and mutual mirror resonance score between self and another agent.

        Args:
            self_potential: Internal macro potential of self.
            other_potential: Internal macro potential of other agent.

        Returns:
            Dictionary with mirror_resonance score and phase_discrepancy.
        """
        if self_potential.shape != other_potential.shape:
            min_len = min(len(self_potential), len(other_potential))
            p1 = self_potential[:min_len]
            p2 = other_potential[:min_len]
        else:
            p1 = self_potential
            p2 = other_potential

        dot_prod = float(np.dot(p1, p2))
        norm1 = float(np.linalg.norm(p1))
        norm2 = float(np.linalg.norm(p2))

        if norm1 == 0.0 or norm2 == 0.0:
            resonance = 0.0
        else:
            resonance = dot_prod / (norm1 * norm2)

        phase_discrepancy = float(np.mean(np.abs(p1 - p2)))

        return {
            "mirror_resonance": resonance,
            "phase_discrepancy": phase_discrepancy,
            "coordination_aligned": resonance > 0.8,
        }
