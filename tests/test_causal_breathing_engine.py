"""
Unit tests for Causal Breathing Engine (core/consciousness/causal_breathing_engine.py)
"""

import numpy as np
import pytest
from core.consciousness.causal_breathing_engine import (
    CausalBreathingEngine,
    MultiDimensionalAttractor,
    VariableResistanceDialMatrix,
    ObserverTopology,
    SpatiotemporalTopologyBuffer,
    ConvergenceResult,
    InhaleResult,
    ExhaleResult
)


def test_attractor_and_dial_matrix():
    cat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    sens = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    morph = np.array([1.1, -0.1, 0.0, 0.0], dtype=np.float32)

    attractor = MultiDimensionalAttractor(
        id="att_apple",
        name="Apple Attractor",
        categorical_vector=cat,
        sensorium_vector=sens,
        morphology_vector=morph
    )

    unified_coord = attractor.get_unified_coordinate()
    assert np.allclose(unified_coord, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-2)

    dial = VariableResistanceDialMatrix(channels=["categorical", "sensorium", "morphology"])
    assert dial.R.shape == (3, 3)
    dial.record_friction("categorical", "sensorium", 0.8)
    assert dial.R[0, 1] > 1.0

    conductance = dial.get_conductance_matrix()
    assert conductance[0, 1] < 1.0

    dial.tune_dials(decay_rate=0.5)
    assert dial.R[0, 1] < 1.5


def test_structural_convergence_sameness_vs_divergence():
    engine = CausalBreathingEngine(convergence_threshold=0.3)

    # Target Attractor (Apple)
    cat_ref = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    sens_ref = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    morph_ref = np.array([0.5, 0.5, 0.0], dtype=np.float32)

    apple_attractor = MultiDimensionalAttractor(
        id="apple",
        name="Apple Object",
        categorical_vector=cat_ref,
        sensorium_vector=sens_ref,
        morphology_vector=morph_ref
    )
    engine.register_attractor(apple_attractor)

    # Input A: Near identical vectors -> "SAMENESS_같다"
    res_same = engine.evaluate_structural_convergence(
        cat_vec=np.array([0.51, 0.49, 0.0], dtype=np.float32),
        sens_vec=np.array([0.49, 0.51, 0.0], dtype=np.float32),
        morph_vec=np.array([0.50, 0.50, 0.01], dtype=np.float32),
        reference_attractor_id="apple"
    )
    assert res_same.is_same is True
    assert res_same.verdict == "SAMENESS_같다"

    # Input B: Divergent vectors -> "DIVERGENCE_다르다"
    res_div = engine.evaluate_structural_convergence(
        cat_vec=np.array([2.5, -1.0, 3.0], dtype=np.float32),
        sens_vec=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        morph_vec=np.array([-2.0, 2.0, 0.0], dtype=np.float32),
        reference_attractor_id="apple"
    )
    assert res_div.is_same is False
    assert res_div.verdict == "DIVERGENCE_다르다"


def test_inhale_tension_and_exhale_pulse():
    engine = CausalBreathingEngine(critical_tension_threshold=8.0)

    # Perform inhales to build up Vt tension
    for i in range(3):
        res = engine.inhale(
            stimulus_id=f"stim_{i}",
            categorical_vector=np.array([1.0, 0.2, 0.0], dtype=np.float32),
            sensorium_vector=np.array([0.8, 0.5, 0.0], dtype=np.float32),
            morphology_vector=np.array([1.2, 0.1, 0.0], dtype=np.float32),
            raw_description=f"Inhale cycle {i}"
        )

    assert engine.current_tension > 0.0

    # Force tension over threshold to trigger EXHALE state
    engine.inhale(
        stimulus_id="stim_heavy",
        categorical_vector=np.array([3.0, 3.0, 3.0], dtype=np.float32),
        sensorium_vector=np.array([-3.0, -3.0, -3.0], dtype=np.float32),
        morphology_vector=np.array([5.0, 0.0, 0.0], dtype=np.float32),
    )

    assert engine.breathing_state == "EXHALE"

    # Exhale with High Abstraction Observer
    obs_high = ObserverTopology(observer_id="researcher", abstraction_capacity=0.9, causal_depth_tolerance=0.8)
    exhale_res_high = engine.exhale(observer=obs_high)

    assert "[High-Tier Structural Pulse]" in exhale_res_high.explanation_pulse
    assert exhale_res_high.released_tension > 0.0
    assert engine.current_tension == 0.0
    assert engine.breathing_state == "INHALE"

    # Build tension again and Exhale with Low Abstraction Observer
    engine.current_tension = 10.0
    obs_low = ObserverTopology(observer_id="novice", abstraction_capacity=0.1, causal_depth_tolerance=0.2)
    exhale_res_low = engine.exhale(observer=obs_low)

    assert "[Concrete Experiential Pulse]" in exhale_res_low.explanation_pulse


def test_spatiotemporal_growth_rings():
    engine = CausalBreathingEngine()

    # Simulate 4 weekly consolidation cycles
    for week in range(4):
        engine.inhale(
            stimulus_id=f"w{week}_stim",
            categorical_vector=np.array([0.1 * week, 0.2, 0.0], dtype=np.float32),
            sensorium_vector=np.array([0.1 * week, 0.2, 0.0], dtype=np.float32),
            morphology_vector=np.array([0.1 * week, 0.2, 0.0], dtype=np.float32),
        )
        res = engine.step_spatiotemporal_cycle(wisdom_summary=f"Weekly wisdom iteration {week}")

    # Check monthly ring creation after 4 weeks
    assert len(engine.spatiotemporal_buffer.weekly_attractors) == 4
    assert len(engine.spatiotemporal_buffer.monthly_rings) == 1

    monthly_ring = engine.spatiotemporal_buffer.monthly_rings[0]
    assert monthly_ring.month_index == 1
    assert "Causal Tectonic Ring 1" in monthly_ring.architectural_summary


def test_introspective_telemetry():
    engine = CausalBreathingEngine()
    engine.inhale(
        stimulus_id="test_stim",
        categorical_vector=np.array([0.1, 0.1, 0.1], dtype=np.float32),
        sensorium_vector=np.array([0.1, 0.1, 0.1], dtype=np.float32),
        morphology_vector=np.array([0.1, 0.1, 0.1], dtype=np.float32),
    )

    telemetry = engine.introspective_telemetry()
    assert "breathing_state" in telemetry
    assert "current_tension_Vt" in telemetry
    assert "dial_resistance_matrix" in telemetry
    assert telemetry["total_inhales"] == 1
