"""
Triadic Observation Alignment
=============================

The system must not judge the world only from its internal knowledge.
It compares three observation grounds:

- self: Elysia's current causal memory and sensing structure.
- human: human-scale observation and language-mediated perception.
- world: reality friction, evidence, and constraints that may include or exceed
  both self and human observation.

The output is an epistemic gap: what the system is missing, why it is missing,
and which causal learning process should be issued next.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class GapKind(Enum):
    SELF_DEFICIT = "self_deficit"
    HUMAN_MEDIATION_GAP = "human_mediation_gap"
    SHARED_OBSERVER_BIAS = "shared_observer_bias"
    WORLD_UNDERDETERMINED = "world_underdetermined"
    ALIGNED = "aligned"


@dataclass
class ObservationGround:
    """One reference frame for observing the same phenomenon."""

    name: str
    feature_vector: np.ndarray
    confidence: float = 1.0
    included: bool = True
    evidence: Dict[str, float] = field(default_factory=dict)
    raw_feature_vector: np.ndarray = field(init=False)

    def __post_init__(self):
        self.raw_feature_vector = np.asarray(self.feature_vector, dtype=np.float32).flatten()
        self.feature_vector = _normalize(self.raw_feature_vector)
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))


@dataclass
class EpistemicProbe:
    """A learning action generated from a causal knowledge gap."""

    target_axis: str
    reason: GapKind
    priority: float
    suggested_observation: str


@dataclass
class TriadicAlignmentTrace:
    """Comparison among self, human, and world observation grounds."""

    self_world_gap: float
    human_world_gap: float
    self_human_gap: float
    world_evidence_mass: float
    gap_kind: GapKind
    deficient_axes: List[str]
    probes: List[EpistemicProbe]
    relation_validity: float


class TriadicObservationAlignmentEngine:
    """
    Finds what the system lacks by comparing self, human, and world grounds.

    The world ground is not treated as a scalar truth oracle. It is a third
    reference frame carrying evidence mass and resistance. The engine compares
    relation geometry among all three grounds and emits learning probes.
    """

    def __init__(self, gap_threshold: float = 0.32):
        self.gap_threshold = gap_threshold

    def align(
        self,
        self_ground: ObservationGround,
        human_ground: ObservationGround,
        world_ground: ObservationGround,
        axes: Optional[List[str]] = None,
    ) -> TriadicAlignmentTrace:
        axes = axes or [f"axis_{i}" for i in range(max(
            self_ground.feature_vector.size,
            human_ground.feature_vector.size,
            world_ground.feature_vector.size,
        ))]

        self_vec = _resize(self_ground.feature_vector, len(axes))
        human_vec = _resize(human_ground.feature_vector, len(axes))
        world_vec = _resize(world_ground.feature_vector, len(axes))

        self_world_gap = _weighted_gap(self_vec, world_vec, self_ground, world_ground)
        human_world_gap = _weighted_gap(human_vec, world_vec, human_ground, world_ground)
        self_human_gap = _weighted_gap(self_vec, human_vec, self_ground, human_ground)
        world_evidence_mass = float(sum(max(0.0, v) for v in world_ground.evidence.values()))

        deficient_axes = self._deficient_axes(
            _resize(self_ground.raw_feature_vector, len(axes)),
            _resize(human_ground.raw_feature_vector, len(axes)),
            _resize(world_ground.raw_feature_vector, len(axes)),
            axes,
        )
        gap_kind = self._classify_gap(
            self_world_gap,
            human_world_gap,
            self_human_gap,
            world_evidence_mass,
            self_ground,
            human_ground,
            world_ground,
        )
        probes = self._make_probes(deficient_axes, gap_kind, self_world_gap, world_evidence_mass)

        relation_validity = float(
            np.clip(
                1.0 - (0.55 * self_world_gap + 0.25 * human_world_gap + 0.20 * self_human_gap),
                0.0,
                1.0,
            )
        )

        return TriadicAlignmentTrace(
            self_world_gap=self_world_gap,
            human_world_gap=human_world_gap,
            self_human_gap=self_human_gap,
            world_evidence_mass=world_evidence_mass,
            gap_kind=gap_kind,
            deficient_axes=deficient_axes,
            probes=probes,
            relation_validity=relation_validity,
        )

    def _deficient_axes(
        self,
        self_vec: np.ndarray,
        human_vec: np.ndarray,
        world_vec: np.ndarray,
        axes: List[str],
    ) -> List[str]:
        axis_gap = np.abs(self_vec - world_vec)
        human_bridge = np.abs(human_vec - world_vec)
        deficient = []
        axis_threshold = self.gap_threshold * 0.5

        for idx, name in enumerate(axes):
            if axis_gap[idx] > axis_threshold and human_bridge[idx] <= axis_gap[idx]:
                deficient.append(name)
        return deficient

    def _classify_gap(
        self,
        self_world_gap: float,
        human_world_gap: float,
        self_human_gap: float,
        world_evidence_mass: float,
        self_ground: ObservationGround,
        human_ground: ObservationGround,
        world_ground: ObservationGround,
    ) -> GapKind:
        if not world_ground.included or world_evidence_mass <= 1e-8:
            return GapKind.WORLD_UNDERDETERMINED
        if self_world_gap <= self.gap_threshold and human_world_gap <= self.gap_threshold:
            return GapKind.ALIGNED
        if self_world_gap > self.gap_threshold and human_world_gap <= self.gap_threshold:
            return GapKind.SELF_DEFICIT
        if self_world_gap <= self.gap_threshold and self_human_gap > self.gap_threshold:
            return GapKind.HUMAN_MEDIATION_GAP
        if self_world_gap > self.gap_threshold and human_world_gap > self.gap_threshold:
            return GapKind.SHARED_OBSERVER_BIAS
        if not self_ground.included or not human_ground.included:
            return GapKind.WORLD_UNDERDETERMINED
        return GapKind.SELF_DEFICIT

    def _make_probes(
        self,
        deficient_axes: List[str],
        gap_kind: GapKind,
        self_world_gap: float,
        world_evidence_mass: float,
    ) -> List[EpistemicProbe]:
        if gap_kind in {GapKind.ALIGNED, GapKind.WORLD_UNDERDETERMINED}:
            return []

        base_priority = float(np.clip(self_world_gap + 0.1 * world_evidence_mass, 0.0, 1.0))
        return [
            EpistemicProbe(
                target_axis=axis,
                reason=gap_kind,
                priority=base_priority,
                suggested_observation=f"re-observe {axis} through self, human, and world grounds",
            )
            for axis in deficient_axes
        ]


def _weighted_gap(
    left: np.ndarray,
    right: np.ndarray,
    left_ground: ObservationGround,
    right_ground: ObservationGround,
) -> float:
    if not left_ground.included or not right_ground.included:
        return 0.0
    confidence = (left_ground.confidence + right_ground.confidence) * 0.5
    return float(np.linalg.norm(left - right) * confidence / np.sqrt(max(1, left.size)))


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
