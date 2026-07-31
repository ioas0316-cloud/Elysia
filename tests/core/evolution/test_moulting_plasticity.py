import pytest
import os
import numpy as np
from core.evolution.moulting_plasticity import MoultingPlasticityEngine
from core.memory.causal_controller import CausalMemoryController


def test_moulting_plasticity_shaping():
    """
    Verifies that MoultingPlasticityEngine can receive non-conforming raw bytes,
    shapes its projection matrix dynamically (Receiver's Plasticity),
    and records physical friction/history without throwing errors.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    engine = MoultingPlasticityEngine(mc, dimensions=3)

    # Initial projection matrix is Identity
    np.testing.assert_array_equal(engine.projection_matrix, np.eye(3, dtype=np.float32))

    # Provide raw byte stimulus
    raw_input = b"Hello, True Dialogue of Empathy"
    res = engine.receive_and_shape(raw_input, modality_hint="empathetic_dialogue")

    assert "tension_vector" in res
    assert "projected_state" in res
    assert "friction" in res
    assert res["moulting_triggered"] is False
    assert res["moulting_count"] == 0

    # Ensure the projection matrix has morphed (Receiver's Plasticity) due to shear stress
    # and is no longer exactly equal to Identity
    assert not np.array_equal(engine.projection_matrix, np.eye(3, dtype=np.float32))


def test_moulting_triggering():
    """
    Verifies that accumulated stress triggers cognitive Moulting (탈피),
    which resets coordinates and records a deep annual ring (나이테) of trauma/friction.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    engine = MoultingPlasticityEngine(mc, dimensions=3)

    # Inject high friction inputs repeatedly to exceed the accumulated friction threshold (3.0)
    high_stress_input = b"\xff\x00\xff\x00\xff\x00\xff\x00\xff\xff"

    moulting_occurred = False
    for _ in range(25):
        res = engine.receive_and_shape(high_stress_input, modality_hint="high_stress")
        if res["moulting_triggered"]:
            moulting_occurred = True
            break

    assert moulting_occurred is True
    assert engine.moulting_count >= 1
    assert "탈피 가동" in res["narrative"]

    # Verify that annual rings (나이테) matrix has accumulated non-zero values
    # representing physical trauma and history
    assert np.any(engine.annual_rings != 0.0)
