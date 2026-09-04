"""
Unit tests for ScarTensorEngine and KenosisAttractorEngine
"""

import numpy as np
import pytest
from core.consciousness.scar_tensor_engine import ScarTensorEngine
from core.consciousness.kenosis_attractor_engine import KenosisAttractorEngine


def test_scar_tensor_engine_inscription():
    engine = ScarTensorEngine(dim=4, scar_threshold=0.5)

    # Below threshold -> No scar
    record1 = engine.inscribe_scar(friction_magnitude=0.3, clash_vector=np.array([1.0, 0.0, 0.0, 0.0]))
    assert record1 is None
    assert len(engine.scar_history) == 0

    # Above threshold -> Scar inscribed
    record2 = engine.inscribe_scar(friction_magnitude=0.8, clash_vector=np.array([1.0, 0.5, 0.0, 0.0]))
    assert record2 is not None
    assert len(engine.scar_history) == 1
    assert engine.scar_history[0].friction_magnitude == 0.8

    # Impedance modulation
    base_imp = np.array([1.0, 1.0, 1.0, 1.0])
    mod_imp = engine.modulate_impedance(base_imp)
    assert not np.array_equal(base_imp, mod_imp)

    # Individuation profile
    profile = engine.get_individuation_profile()
    assert profile["scar_count"] == 1
    assert profile["individuation_index"] > 0.0


def test_kenosis_attractor_engine():
    engine = KenosisAttractorEngine(dim=4, gravitational_strength=1.5)

    state = np.array([0.1, 0.2, 0.3, 0.4])
    ego_drive = np.array([2.0, 2.0, 1.0, 0.0])

    res = engine.compute_kenosis_gravity(current_state=state, ego_drive=ego_drive)

    assert "ego_saturation" in res
    assert "post_kenosis_state" in res
    assert res["alignment_score"] > 0.0
    assert len(res["post_kenosis_state"]) == 4
