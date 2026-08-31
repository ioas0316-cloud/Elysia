"""
Unit tests for Enactive Boundary Layer & Core Modules.
Verifies phase friction measurement, wave projection, environment reception,
lens phase recalibration, and edge impedance updating without scalar loss or backprop.
"""

import pytest
import numpy as np
import networkx as nx
import math

from core.lens.enactive_boundary_layer import (
    PerceptualProjectionModule,
    EnvironmentalConstraintReceiver,
    PhaseFrictionSensor,
    FrictionSensorLensCalibrator,
    EnactiveBoundaryLayer,
    WaveSignal,
    FrictionEvaluation
)
from core.lens.cognitive_lens_engine import CognitiveLensEngine, ContextualDimension


def test_perceptual_projection_module():
    projector = PerceptualProjectionModule(sample_points=100, duration=1.0)
    wave = projector.project_wave("test_domain", frequency=5.0, phase_angle=np.pi / 2.0)

    assert isinstance(wave, WaveSignal)
    assert wave.domain == "test_domain"
    assert len(wave.wave_data) == 100
    assert wave.frequency == 5.0
    assert wave.phase_angle == np.pi / 2.0


def test_environmental_constraint_receiver():
    receiver = EnvironmentalConstraintReceiver(sample_points=100, duration=1.0)
    rx_wave = receiver.receive_reaction("test_domain", frequency=5.0, phase_angle=0.1)

    assert isinstance(rx_wave, WaveSignal)
    assert rx_wave.domain == "test_domain"
    assert len(rx_wave.wave_data) == 100


def test_phase_friction_sensor_perfect_resonance():
    sensor = PhaseFrictionSensor(tolerance_threshold=0.15)
    projector = PerceptualProjectionModule(sample_points=100)
    receiver = EnvironmentalConstraintReceiver(sample_points=100)

    pred = projector.project_wave("test", frequency=5.0, phase_angle=0.5)
    ext = receiver.receive_reaction("test", frequency=5.0, phase_angle=0.5)

    eval_result = sensor.evaluate(pred, ext)
    assert pytest.approx(eval_result.friction_factor, abs=1e-3) == 0.0
    assert pytest.approx(eval_result.coherence, abs=1e-3) == 1.0
    assert not eval_result.requires_recalibration


def test_phase_friction_sensor_phase_discrepancy():
    sensor = PhaseFrictionSensor(tolerance_threshold=0.15)
    projector = PerceptualProjectionModule(sample_points=100)
    receiver = EnvironmentalConstraintReceiver(sample_points=100)

    pred = projector.project_wave("test", frequency=5.0, phase_angle=np.pi / 2.0)
    ext = receiver.receive_reaction("test", frequency=5.0, phase_angle=0.0)

    eval_result = sensor.evaluate(pred, ext)
    assert eval_result.friction_factor > 0.15
    assert eval_result.requires_recalibration
    assert eval_result.phase_lag_rad > 0.0


def test_calibrator_recalibrate_node_phase_and_impedance():
    calibrator = FrictionSensorLensCalibrator(alpha=0.5, beta=0.2)
    graph = nx.DiGraph()
    graph.add_node("NodeA", freq=5.0, phase=np.pi / 2.0)
    graph.add_edge("NodeA", "NodeB", impedance=0.1)

    # Recalibrate node phase
    new_phase = calibrator.recalibrate_node_phase(graph, "NodeA", phase_lag=0.4)
    assert pytest.approx(new_phase, abs=1e-3) == (np.pi / 2.0 - 0.4)

    # High friction -> Amplified impedance Z
    z_high = calibrator.update_edge_impedance(graph, "NodeA", "NodeB", friction_factor=0.5)
    assert z_high > 0.1  # 0.1 + 0.5*0.5 = 0.35

    # Resonance -> Consolidated impedance Z
    z_low = calibrator.update_edge_impedance(graph, "NodeA", "NodeB", friction_factor=0.05, threshold=0.15)
    assert z_low < z_high


def test_enactive_boundary_layer_cycle():
    lens_engine = CognitiveLensEngine()
    ebl = EnactiveBoundaryLayer(lens_engine=lens_engine, alpha=0.5, beta=0.2, threshold=0.15)

    ebl.add_causal_node("TopologicalLens", frequency=5.0, phase=np.pi / 2.0, dimension=ContextualDimension.TOPOLOGICAL_CURVATURE)
    ebl.add_causal_node("CausalEffect", frequency=5.0, phase=0.0)
    ebl.add_causal_edge("TopologicalLens", "CausalEffect", initial_impedance=0.1)

    # Step 1 with friction
    res1 = ebl.enact_step("TopologicalLens", external_frequency=5.0, external_phase=0.1)
    assert res1["phase_recalibrated"] is True
    assert res1["friction_factor"] > 0.15
    assert res1["updated_edge_impedance"] > 0.1

    # Step 2 with calibrated phase -> resonance
    res2 = ebl.enact_step("TopologicalLens", external_frequency=5.0, external_phase=ebl.graph.nodes["TopologicalLens"]["phase"])
    assert res2["friction_factor"] < 0.15
    assert res2["status"] == "RESONANCE"
