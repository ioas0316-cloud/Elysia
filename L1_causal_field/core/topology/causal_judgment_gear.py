"""
Causal Judgment Gear
====================

Judgment is modeled as cause-process-result meshing, not as a class label.
When the cause, process, or result changes, the change must be justified by
the relation and connectivity shifts inside the same causal gear chain.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from core.topology.informational_phase_observation import ChromaticVector


class GearRegion(Enum):
    CAUSE = "cause"
    PROCESS = "process"
    RESULT = "result"


@dataclass
class CausalGearState:
    """
    One gear in a cause-process-result chain.

    - vector: causal direction and magnitude.
    - resistance: variable-resistor dial for friction.
    - conductance: how easily impact transfers to the next gear.
    - chromatic: Flux/Order/Entropy signature.
    """

    region: GearRegion
    name: str
    vector: np.ndarray
    resistance: float = 0.5
    conductance: float = 0.5
    chromatic: Optional[ChromaticVector] = None

    def __post_init__(self):
        self.vector = _normalize(np.asarray(self.vector, dtype=np.float32).flatten())
        self.resistance = float(np.clip(self.resistance, 0.0, 1.0))
        self.conductance = float(np.clip(self.conductance, 0.0, 1.0))
        if self.chromatic is None:
            self.chromatic = ChromaticVector(flux=1.0, order=1.0, entropy=0.2)


@dataclass
class GearMesh:
    """Continuous meshing quality between two adjacent causal gears."""

    source: GearRegion
    target: GearRegion
    alignment: float
    impedance_gap: float
    chromatic_friction: float
    transfer_validity: float


@dataclass
class CausalJudgmentTrace:
    """Reason/evidence carried by the cause-process-result structure itself."""

    cause_to_process: GearMesh
    process_to_result: GearMesh
    structural_validity: float
    causal_tension: float
    judgment_ground: str
    changed_region: Optional[GearRegion] = None
    change_justification: Optional[Dict[str, float]] = None


class CausalJudgmentGear:
    """
    Evaluates why a result follows from a cause through a process.

    The answer is not selected externally. It is derived from the meshing
    of relation alignment, conductance/resistance, and chromatic continuity.
    """

    def evaluate(
        self,
        cause: CausalGearState,
        process: CausalGearState,
        result: CausalGearState,
    ) -> CausalJudgmentTrace:
        cause_to_process = self._mesh(cause, process)
        process_to_result = self._mesh(process, result)

        structural_validity = float(
            np.sqrt(cause_to_process.transfer_validity * process_to_result.transfer_validity)
        )
        causal_tension = float(
            1.0 - structural_validity
            + 0.5 * (cause_to_process.impedance_gap + process_to_result.impedance_gap)
        )

        return CausalJudgmentTrace(
            cause_to_process=cause_to_process,
            process_to_result=process_to_result,
            structural_validity=float(np.clip(structural_validity, 0.0, 1.0)),
            causal_tension=float(np.clip(causal_tension, 0.0, 2.0)),
            judgment_ground=self._ground_text(structural_validity, causal_tension),
        )

    def compare_change(
        self,
        baseline: CausalJudgmentTrace,
        changed: CausalJudgmentTrace,
        changed_region: GearRegion,
    ) -> CausalJudgmentTrace:
        """
        Explains why a changed cause/process/result produced a changed judgment.

        The justification is expressed as deltas in the gear relations rather
        than as a detached explanation string.
        """
        cp_delta = changed.cause_to_process.transfer_validity - baseline.cause_to_process.transfer_validity
        pr_delta = changed.process_to_result.transfer_validity - baseline.process_to_result.transfer_validity
        validity_delta = changed.structural_validity - baseline.structural_validity
        tension_delta = changed.causal_tension - baseline.causal_tension

        changed.changed_region = changed_region
        changed.change_justification = {
            "cause_process_transfer_delta": float(cp_delta),
            "process_result_transfer_delta": float(pr_delta),
            "structural_validity_delta": float(validity_delta),
            "causal_tension_delta": float(tension_delta),
            "relation_continuity_evidence": float(1.0 - min(1.0, abs(cp_delta - pr_delta))),
        }
        return changed

    def _mesh(self, source: CausalGearState, target: CausalGearState) -> GearMesh:
        source_vec = _resize(source.vector, max(source.vector.size, target.vector.size))
        target_vec = _resize(target.vector, source_vec.size)
        alignment = float(np.clip(np.dot(source_vec, target_vec), -1.0, 1.0))

        source_impedance = source.resistance / (source.conductance + 1e-8)
        target_impedance = target.resistance / (target.conductance + 1e-8)
        impedance_gap = float(abs(source_impedance - target_impedance) / (1.0 + source_impedance + target_impedance))

        chroma_source = source.chromatic.to_array()
        chroma_target = target.chromatic.to_array()
        chromatic_friction = float(np.linalg.norm(chroma_source - chroma_target) / np.sqrt(3.0))

        positive_alignment = max(0.0, alignment)
        continuity = 1.0 / (1.0 + impedance_gap + chromatic_friction)
        transfer_validity = float(
            np.clip(positive_alignment * continuity * source.conductance * (1.0 - target.resistance * 0.5), 0.0, 1.0)
        )

        return GearMesh(
            source=source.region,
            target=target.region,
            alignment=alignment,
            impedance_gap=impedance_gap,
            chromatic_friction=chromatic_friction,
            transfer_validity=transfer_validity,
        )

    def _ground_text(self, structural_validity: float, causal_tension: float) -> str:
        validity_words = [
            (0.75, "strongly grounded"),
            (0.45, "partially grounded"),
            (0.0, "weakly grounded"),
        ]
        tension_words = [
            (0.85, "high transformation tension"),
            (0.35, "moderate transformation tension"),
            (0.0, "low transformation tension"),
        ]
        validity = next(text for threshold, text in validity_words if structural_validity >= threshold)
        tension = next(text for threshold, text in tension_words if causal_tension >= threshold)
        return f"{validity}; {tension}"


def _resize(vector: np.ndarray, target_size: int) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).flatten()
    if arr.size == target_size:
        return arr
    if arr.size == 0:
        return np.zeros(target_size, dtype=np.float32)
    indices = np.linspace(0, arr.size - 1, target_size)
    return np.interp(indices, np.arange(arr.size), arr).astype(np.float32)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-8:
        return vector
    return vector / norm
