import pytest
import numpy as np
from synaptic_architecture.symbol_generative_causality_engine import GenerativeCausalityEngine


def test_symbol_deconstruction():
    engine = GenerativeCausalityEngine(dim=3)

    # Test deconstruction of '100°C'
    decon_100c = engine.deconstruct_symbol("100°C")
    assert decon_100c["symbol"] == "100°C"
    assert decon_100c["category"] == "measurement"
    assert decon_100c["deconstructed_dynamics"]["phase_transition_energy"] == 100.0
    assert len(decon_100c["deconstructed_dynamics"]["friction_eigenvalues"]) == 3

    # Test deconstruction of 'v = d / t'
    decon_vel = engine.deconstruct_symbol("v = d / t")
    assert decon_vel["symbol"] == "v = d / t"
    assert decon_vel["deconstructed_dynamics"]["spatial_resistance"] == 10.0
    assert decon_vel["deconstructed_dynamics"]["time_delay"] == 2.0
    assert abs(decon_vel["deconstructed_dynamics"]["causal_ratio_d_over_t"] - 5.0) < 1e-5


def test_active_probing():
    engine = GenerativeCausalityEngine(dim=3)

    probing_force = np.array([1.0, 2.0, 0.0])
    external_reaction = np.array([0.5, 1.0, 0.0])

    res = engine.active_probe_environment(probing_force, external_reaction)

    assert "measured_resistance" in res
    assert "measured_delay" in res
    assert res["probing_force_norm"] > 0
    assert len(res["friction_tensor"]) == 3


def test_rediscover_generative_causality():
    engine = GenerativeCausalityEngine(dim=3)

    probing_force = np.array([1.0, 0.0, 0.0])
    external_reaction = np.array([0.2, 0.0, 0.0])

    observed = engine.active_probe_environment(probing_force, external_reaction)
    blueprint = engine.rediscover_generative_causality("v = d / t", observed)

    assert blueprint["symbol"] == "v = d / t"
    assert blueprint["resonance_degree"] > 0.0
    assert "Generative Causality" in blueprint["origin_explanation"]


def test_4stage_cognition():
    engine = GenerativeCausalityEngine(dim=3)

    test_action = np.array([1.0, 0.5, 0.2])
    res = engine.execute_4stage_cognition("100°C", test_action)

    assert res["symbol"] == "100°C"
    assert res["cognition"]["stage"] == "1. Cognition"
    assert res["thought"]["stage"] == "2. Thought"
    assert res["judgment"]["stage"] == "3. Judgment"
    assert res["discrimination"]["mode"] == "Generative Causality (Living Geometry)"
