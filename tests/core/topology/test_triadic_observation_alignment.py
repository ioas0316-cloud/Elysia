"""
Tests for self-human-world observation triangulation.
"""

import numpy as np

from core.topology.triadic_observation_alignment import (
    GapKind,
    ObservationGround,
    TriadicObservationAlignmentEngine,
)


AXES = ["color", "growth", "chemistry", "edibility"]


def _ground(name, values, confidence=1.0, included=True, evidence=None):
    return ObservationGround(
        name=name,
        feature_vector=np.array(values, dtype=np.float32),
        confidence=confidence,
        included=included,
        evidence=evidence or {},
    )


def test_self_deficit_when_human_and_world_align_against_internal_memory():
    engine = TriadicObservationAlignmentEngine(gap_threshold=0.22)
    self_ground = _ground("self", [0.1, 0.8, 0.8, 0.8])
    human_ground = _ground("human", [0.9, 0.8, 0.8, 0.8])
    world_ground = _ground("world", [0.95, 0.8, 0.8, 0.8], evidence={"orchard_sample": 0.9})

    trace = engine.align(self_ground, human_ground, world_ground, AXES)

    assert trace.gap_kind is GapKind.SELF_DEFICIT
    assert "color" in trace.deficient_axes
    assert trace.probes[0].target_axis == "color"
    assert trace.self_world_gap > trace.human_world_gap


def test_shared_observer_bias_when_self_and_human_both_disagree_with_world():
    engine = TriadicObservationAlignmentEngine(gap_threshold=0.22)
    self_ground = _ground("self", [0.2, 0.9, 0.2, 0.8])
    human_ground = _ground("human", [0.25, 0.9, 0.2, 0.8])
    world_ground = _ground("world", [0.9, 0.9, 0.85, 0.8], evidence={"chemical_assay": 1.0})

    trace = engine.align(self_ground, human_ground, world_ground, AXES)

    assert trace.gap_kind is GapKind.SHARED_OBSERVER_BIAS
    assert "color" in trace.deficient_axes
    assert "chemistry" in trace.deficient_axes
    assert trace.relation_validity < 1.0


def test_world_underdetermined_when_reality_ground_is_excluded_or_unsupported():
    engine = TriadicObservationAlignmentEngine(gap_threshold=0.22)
    self_ground = _ground("self", [0.9, 0.8, 0.8, 0.8])
    human_ground = _ground("human", [0.2, 0.8, 0.8, 0.8])
    world_ground = _ground("world", [0.0, 0.0, 0.0, 0.0], included=False)

    trace = engine.align(self_ground, human_ground, world_ground, AXES)

    assert trace.gap_kind is GapKind.WORLD_UNDERDETERMINED
    assert trace.probes == []


def test_aligned_when_all_three_observation_grounds_resonate():
    engine = TriadicObservationAlignmentEngine(gap_threshold=0.22)
    self_ground = _ground("self", [0.8, 0.7, 0.9, 0.9])
    human_ground = _ground("human", [0.82, 0.68, 0.88, 0.9])
    world_ground = _ground("world", [0.81, 0.7, 0.89, 0.92], evidence={"direct_observation": 1.0})

    trace = engine.align(self_ground, human_ground, world_ground, AXES)

    assert trace.gap_kind is GapKind.ALIGNED
    assert trace.deficient_axes == []
    assert trace.relation_validity > 0.8
