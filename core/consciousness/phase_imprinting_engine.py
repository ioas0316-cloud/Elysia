"""
Phase Imprinting & Hierarchical Grounding Engine (위상 각인 및 계층적 접지 엔진)

Transforms observed triune spectrum waveforms into higher-order semantic definition structures
(Hierarchical Grounding) and recalibrates the cognitive lens itself upon causal outcome feedback
(Phase Imprinting), storing experience as active cognitive recalibration rather than dead logs.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from core.lens.triune_awareness_lens import TriuneAwarenessLens, TriuneSpectrum


@dataclass
class SemanticDefinitionNode:
    """
    Represents a higher-level semantic definition grounded in a triune spectrum signature.
    """
    name: str
    target_spectrum: Tuple[float, float, float]  # (cycle, amplitude, phase)
    resonance_threshold: float = 0.7
    grounded_causal_meaning: str = ""
    associated_resistance_delta: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    imprint_count: int = 0

    def match_resonance(self, spectrum: TriuneSpectrum) -> float:
        """
        Calculates how closely a spectrum signature matches this grounded semantic definition.
        """
        c_diff = abs(spectrum.cycle - self.target_spectrum[0])
        a_diff = abs(spectrum.amplitude - self.target_spectrum[1])

        # Angular distance for phase
        p_diff = abs((spectrum.phase - self.target_spectrum[2] + np.pi) % (2 * np.pi) - np.pi)

        distance = np.sqrt(c_diff**2 + a_diff**2 + p_diff**2)
        affinity = 1.0 / (1.0 + distance)
        return float(affinity)


class PhaseImprintingEngine:
    """
    Phase Imprinting & Hierarchical Grounding Engine

    Acts as the cognitive bridge between the 3D Awareness Lens and high-level semantic reasoning.
    """

    def __init__(
        self,
        lens: TriuneAwarenessLens,
        recalibration_rate: float = 0.1,
    ):
        self.lens = lens
        self.recalibration_rate = recalibration_rate
        self.grounded_semantic_registry: Dict[str, SemanticDefinitionNode] = {}
        self.imprint_history: List[Dict[str, Any]] = []

    def register_semantic_definition(
        self,
        name: str,
        target_spectrum: Tuple[float, float, float],
        causal_meaning: str = "",
        resistance_delta: Optional[np.ndarray] = None,
    ) -> SemanticDefinitionNode:
        """
        Grounds a new higher-level semantic definition onto a specific spectrum signature.
        """
        if resistance_delta is None:
            resistance_delta = np.zeros((3, 3))

        node = SemanticDefinitionNode(
            name=name,
            target_spectrum=target_spectrum,
            grounded_causal_meaning=causal_meaning,
            associated_resistance_delta=resistance_delta,
        )
        self.grounded_semantic_registry[name] = node
        return node

    def Ground_hierarchical_definition(
        self,
        spectrum: TriuneSpectrum,
    ) -> Tuple[Optional[SemanticDefinitionNode], float]:
        """
        Grounds an observed triune spectrum onto the most resonant higher-level semantic definition.
        """
        if not self.grounded_semantic_registry:
            return None, 0.0

        best_node = None
        best_affinity = -1.0

        for node in self.grounded_semantic_registry.values():
            affinity = node.match_resonance(spectrum)
            if affinity > best_affinity:
                best_affinity = affinity
                best_node = node

        return best_node, best_affinity

    def imprint_phase_experience(
        self,
        spectrum: TriuneSpectrum,
        outcome_tension: float,
        causal_feedback_phase: float,
        associated_semantic_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Performs Phase Imprinting (경험적 인과 각인 및 렌즈 재조율):
        Recalibrates the internal variable resistance, intrinsic phase, and phase coupling
        of the TriuneAwarenessLens based on actual experience / tension feedback.

        Args:
            spectrum: The initial spectrum observed before action/interaction.
            outcome_tension: The resulting resistance/tension experienced after action.
            causal_feedback_phase: The phase returned by the environment/field.
            associated_semantic_name: Optional semantic node being updated.

        Returns:
            Dictionary containing recalibration diagnostics.
        """
        old_resistance = self.lens.resistance_matrix.copy()
        old_intrinsic_phase = self.lens.intrinsic_phase
        old_coupling = self.lens.phase_coupling_strength

        # 1. Recalibrate Variable Resistance Matrix based on outcome tension
        # Tension indicates where internal friction was under- or over-estimated
        tension_error = outcome_tension - spectrum.amplitude
        resistance_adjustment = self.recalibration_rate * tension_error * np.eye(3)
        self.lens.resistance_matrix += resistance_adjustment
        # Ensure positive definite resistance
        self.lens.resistance_matrix = np.maximum(self.lens.resistance_matrix, 0.1)

        # 2. Recalibrate Intrinsic Phase based on causal feedback
        phase_error = (causal_feedback_phase - spectrum.phase + np.pi) % (2 * np.pi) - np.pi
        self.lens.intrinsic_phase = (self.lens.intrinsic_phase + self.recalibration_rate * phase_error) % (2 * np.pi)

        # 3. Recalibrate Phase Coupling Strength
        # High resonance increases coupling strength, interference reduces it
        coupling_delta = self.recalibration_rate * (spectrum.resonance - spectrum.interference)
        self.lens.phase_coupling_strength = float(
            np.clip(self.lens.phase_coupling_strength + coupling_delta, 0.1, 5.0)
        )

        # 4. Update semantic definition node if attached
        if associated_semantic_name and associated_semantic_name in self.grounded_semantic_registry:
            node = self.grounded_semantic_registry[associated_semantic_name]
            node.imprint_count += 1
            # Shift target spectrum slightly towards experienced phase & cycle
            new_cycle = 0.9 * node.target_spectrum[0] + 0.1 * spectrum.cycle
            new_amp = 0.9 * node.target_spectrum[1] + 0.1 * outcome_tension
            new_phase = 0.9 * node.target_spectrum[2] + 0.1 * causal_feedback_phase
            node.target_spectrum = (new_cycle, new_amp, new_phase)

        imprint_record = {
            "initial_spectrum": spectrum.to_dict(),
            "outcome_tension": float(outcome_tension),
            "causal_feedback_phase": float(causal_feedback_phase),
            "resistance_delta": (self.lens.resistance_matrix - old_resistance).tolist(),
            "phase_shift": float(self.lens.intrinsic_phase - old_intrinsic_phase),
            "coupling_shift": float(self.lens.phase_coupling_strength - old_coupling),
            "associated_semantic": associated_semantic_name,
        }

        self.imprint_history.append(imprint_record)
        return imprint_record
