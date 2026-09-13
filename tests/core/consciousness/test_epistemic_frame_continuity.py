import pytest
import numpy as np
import time
from core.consciousness.epistemic_frame_continuity_engine import (
    EpistemicFrameContinuityEngine,
    SpatiotemporalEmergenceTracker,
    CausalVoidHealingEngine,
    VariationalFreeEnergyCalculator,
    StructuralPlasticityAdapter
)

def test_spatiotemporal_emergence():
    tracker = SpatiotemporalEmergenceTracker(dimension=16)
    s1 = np.ones(16, dtype=np.float64) * 0.1
    s2 = np.ones(16, dtype=np.float64) * 0.5

    t0 = 1000.0
    res1 = tracker.update_frame(s1, current_time=t0)
    assert res1["status"] == "INITIAL_FRAME_ESTABLISHED"

    res2 = tracker.update_frame(s2, current_time=t0 + 0.1)
    assert res2["status"] == "CONTINUOUS_FLOW"
    assert res2["velocity_norm"] > 0
    assert tracker.accumulated_causal_time > 0
    assert res2["spatial_distance"] > 0

def test_causal_void_healing_nan():
    dimension = 16
    tracker = SpatiotemporalEmergenceTracker(dimension=dimension)
    healer = CausalVoidHealingEngine(dimension=dimension)

    # Establish valid baseline state & momentum
    s1 = np.ones(dimension, dtype=np.float64) * 0.2
    s2 = np.ones(dimension, dtype=np.float64) * 0.3
    tracker.update_frame(s1, current_time=1.0)
    tracker.update_frame(s2, current_time=1.1)

    # Create NaN corrupted state
    nan_state = np.ones(dimension, dtype=np.float64) * 0.5
    nan_state[3] = np.nan
    nan_state[7] = np.nan

    valid_state, heal_info = healer.detect_and_heal_void(nan_state, tracker, causal_tension=0.0)

    assert heal_info["is_void"] is True
    assert heal_info["healed"] is True
    assert not np.isnan(valid_state).any()
    assert healer.void_count == 1

def test_vfe_calculator_and_plasticity():
    dimension = 16
    vfe_calc = VariationalFreeEnergyCalculator(dimension=dimension)
    plasticity = StructuralPlasticityAdapter(dimension=dimension)

    s_pred = np.zeros(dimension, dtype=np.float64)
    s_pred[0] = 1.0

    s_act = np.zeros(dimension, dtype=np.float64)
    s_act[0] = 0.8
    s_act[1] = 0.2

    vfe_info = vfe_calc.compute_vfe(s_pred, s_act, internal_tension=2.0)
    assert vfe_info["variational_free_energy"] > 0

    heal_info = {"healed": False}
    adapt_info = plasticity.adapt_structure(vfe_info, heal_info)
    assert adapt_info["status"] == "PLASTICITY_ADAPTED"
    assert plasticity.total_plastic_adaptations == 1

def test_epistemic_frame_continuity_engine_full_pipeline():
    engine = EpistemicFrameContinuityEngine(dimension=16)

    # 1. First frame
    s1 = np.random.randn(16)
    r1 = engine.process_frame(s1, timestamp=1.0)
    assert r1["status"] == "FRAME_CONTINUITY_PROCESSED"

    # 2. Second frame (continuous transition)
    s2 = s1 + 0.1 * np.random.randn(16)
    r2 = engine.process_frame(s2, timestamp=1.1)
    assert r2["spatiotemporal_info"]["velocity_norm"] > 0
    assert r2["vfe_info"]["variational_free_energy"] >= 0

    # 3. Third frame with NaN corruption (void)
    s3_nan = s2.copy()
    s3_nan[0] = np.nan
    s3_nan[5] = np.nan
    r3 = engine.process_frame(s3_nan, timestamp=1.2)
    assert r3["heal_info"]["is_void"] is True
    assert not np.isnan(r3["valid_state"]).any()

    summary = engine.get_continuity_summary()
    assert summary["total_frames_processed"] == 3
    assert summary["void_count"] == 1
    assert summary["accumulated_causal_time"] > 0
