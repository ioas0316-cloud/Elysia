"""
Tests for cause-process-result judgment as causal gear meshing.
"""

import numpy as np

from core.topology.causal_judgment_gear import (
    CausalGearState,
    CausalJudgmentGear,
    GearRegion,
)
from core.topology.informational_phase_observation import ChromaticVector


def _state(region, name, vector, resistance=0.25, conductance=0.85, chromatic=None):
    return CausalGearState(
        region=region,
        name=name,
        vector=np.array(vector, dtype=np.float32),
        resistance=resistance,
        conductance=conductance,
        chromatic=chromatic or ChromaticVector(flux=1.0, order=1.0, entropy=0.2),
    )


def test_judgment_is_grounded_by_cause_process_result_meshing():
    gear = CausalJudgmentGear()
    cause = _state(GearRegion.CAUSE, "heat_input", [1.0, 0.2, 0.0])
    process = _state(GearRegion.PROCESS, "phase_transition", [0.9, 0.3, 0.1])
    result = _state(GearRegion.RESULT, "expanded_volume", [0.8, 0.4, 0.1])

    trace = gear.evaluate(cause, process, result)

    assert trace.cause_to_process.transfer_validity > 0.0
    assert trace.process_to_result.transfer_validity > 0.0
    assert trace.structural_validity > 0.0
    assert "grounded" in trace.judgment_ground


def test_changed_result_is_explained_by_process_result_relation_delta():
    gear = CausalJudgmentGear()
    cause = _state(GearRegion.CAUSE, "heat_input", [1.0, 0.2, 0.0])
    process = _state(GearRegion.PROCESS, "phase_transition", [0.9, 0.3, 0.1])
    result = _state(GearRegion.RESULT, "expanded_volume", [0.8, 0.4, 0.1])
    changed_result = _state(
        GearRegion.RESULT,
        "contracted_volume",
        [-0.8, -0.4, -0.1],
        resistance=0.7,
        conductance=0.35,
        chromatic=ChromaticVector(flux=0.4, order=0.6, entropy=1.2),
    )

    baseline = gear.evaluate(cause, process, result)
    changed = gear.evaluate(cause, process, changed_result)
    explained = gear.compare_change(baseline, changed, GearRegion.RESULT)

    assert explained.changed_region is GearRegion.RESULT
    assert explained.change_justification is not None
    assert explained.change_justification["process_result_transfer_delta"] < 0.0
    assert explained.change_justification["structural_validity_delta"] < 0.0
    assert explained.causal_tension > baseline.causal_tension


def test_changed_process_preserves_evidence_as_relation_continuity():
    gear = CausalJudgmentGear()
    cause = _state(GearRegion.CAUSE, "premise", [0.2, 1.0, 0.2])
    process = _state(GearRegion.PROCESS, "valid_operator", [0.25, 0.95, 0.2])
    result = _state(GearRegion.RESULT, "conclusion", [0.3, 0.9, 0.25])
    changed_process = _state(
        GearRegion.PROCESS,
        "distorted_operator",
        [0.9, -0.9, 0.1],
        resistance=0.8,
        conductance=0.25,
    )

    baseline = gear.evaluate(cause, process, result)
    changed = gear.evaluate(cause, changed_process, result)
    explained = gear.compare_change(baseline, changed, GearRegion.PROCESS)

    assert explained.changed_region is GearRegion.PROCESS
    assert 0.0 <= explained.change_justification["relation_continuity_evidence"] <= 1.0
    assert explained.change_justification["causal_tension_delta"] > 0.0
