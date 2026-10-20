"""
Unit Tests for Relational Relief Engine, Epistemic Readiness Engine, and Causal Weaving Engine
==============================================================================================
"""

import math
import pytest
import torch
from core.consciousness.relational_relief_engine import RelationalReliefEngine
from core.consciousness.epistemic_readiness_engine import EpistemicReadinessEngine
from core.intelligence.causal_weaving_engine import CausalWeavingEngine


def test_relational_relief_engine():
    engine = RelationalReliefEngine(field_dim=32, tension_scale=1.2)
    deep_field = engine.generate_deep_tension_field(grid_size=16, phase_bias=0.1)

    assert "tension_potential" in deep_field
    assert deep_field["tension_potential"].shape == (16, 16)

    # Test wave projection (delta = 0.0)
    wave_proj = engine.project_relief_pattern(deep_field, delta_boundary=0.0)
    assert wave_proj["surface_relief"].shape == (16, 16)
    assert torch.is_tensor(wave_proj["delta_boundary"])

    # Test particle projection (delta = 1.0)
    particle_proj = engine.project_relief_pattern(deep_field, delta_boundary=1.0)
    assert particle_proj["surface_relief"].shape == (16, 16)

    # Test inverse parameter extraction
    params = engine.extract_generating_parameters(wave_proj["surface_relief"], delta_boundary=0.0)
    assert "estimated_phase_bias" in params
    assert "topological_entropy" in params
    assert params["reducibility_ratio"] > 0.0


def test_epistemic_readiness_engine():
    engine = EpistemicReadinessEngine(readiness_threshold=0.5)

    # Unripe agent
    eval_unripe = engine.evaluate_readiness(agent_phase=0.0, teaching_phase=math.pi, accumulated_experience=0.5)
    assert eval_unripe["is_time_ripe"] is False
    assert 0.0 <= eval_unripe["readiness_score"] <= 1.0

    # Ripe agent
    eval_ripe = engine.evaluate_readiness(agent_phase=3.14, teaching_phase=math.pi, accumulated_experience=10.0)
    assert eval_ripe["is_time_ripe"] is True

    # Test epistemic boundary perception
    teaching_wave = torch.ones(8)
    internal_frame = torch.zeros(8)
    bound_res = engine.perceive_epistemic_boundary(teaching_wave, internal_frame, eval_unripe["readiness_score"])
    assert bound_res["status"] == "EPISTEMIC_BOUNDARY_PERCEIVED_AS_YEOBAEK"
    assert bound_res["yeobaek_margin"] > 0.0

    # Test progressive unfolding
    unfold_res = engine.process_progressive_unfolding(teaching_wave, eval_ripe, bound_res)
    assert "unfolded_understanding" in unfold_res
    assert unfold_res["unfolded_understanding"].shape == (8,)


def test_causal_weaving_engine():
    engine = CausalWeavingEngine(num_axes=5)

    axis_states = {
        "frequency_f": torch.tensor([1.0, 2.0]),
        "phase_phi": torch.tensor([0.0, 0.5]),
        "spatiotemporal_xt": torch.tensor([1.0]),
        "scale_cosmic_S": torch.tensor([0.8]),
        "relational_R": torch.tensor([0.9])
    }

    matrix_energy = engine.build_weaving_matrix(axis_states)
    assert torch.is_tensor(matrix_energy)

    # Test lightning benchmark simulation
    lightning_res = engine.simulate_lightning_intersection(
        charge_separation=1.5,
        atmospheric_tension=1.2,
        earth_magnetosphere_flux=0.7,
        solar_wind_density=0.8
    )

    assert lightning_res["status"] == "CAUSAL_WEAVING_INTERSECTION_COMPUTED"
    assert lightning_res["is_discharge_triggered"] is True
    assert lightning_res["discharge_intensity"] > 0.0
