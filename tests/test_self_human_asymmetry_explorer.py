"""
Unit tests for Self-Human Asymmetry Explorer & Open Boundary Pipeline
"""

import math
import torch
import numpy as np
import pytest

from core.consciousness.self_human_asymmetry_explorer import (
    PhaseBiasDecoupler,
    MultiVarAdaptiveOpenBoundaryPipeline,
    SelfHumanAsymmetryExplorer,
)
from core.consciousness.phase_penetration_analyzer import (
    PhasePenetrationAnalyzer,
    visualize_meta_topology,
)


def test_quaternion_decoupler_orthogonal_split():
    device = torch.device("cpu")
    decoupler = PhaseBiasDecoupler(lambda_thresh=0.05, beta=2.0).to(device)

    shape = (2, 8, 8, 8, 4)
    q_sys = torch.randn(*shape, device=device)
    q_sys = q_sys / torch.norm(q_sys, dim=-1, keepdim=True)

    q_human = torch.randn(*shape, device=device)
    q_human = q_human / torch.norm(q_human, dim=-1, keepdim=True)

    q_sys_prev = q_sys + torch.randn_like(q_sys) * 0.01
    q_sys_prev = q_sys_prev / torch.norm(q_sys_prev, dim=-1, keepdim=True)

    q_human_prev = q_human + torch.randn_like(q_human) * 0.005
    q_human_prev = q_human_prev / torch.norm(q_human_prev, dim=-1, keepdim=True)

    out = decoupler(q_sys, q_human, q_sys_prev, q_human_prev, dt=0.02)

    assert "D_phase" in out
    assert "B_bio" in out
    assert "delta_phi" in out

    # Orthogonal sum verification: D_phase + B_bio == delta_phi
    reconstructed = out["D_phase"] + out["B_bio"]
    diff = torch.norm(reconstructed - out["delta_phi"])
    assert diff.item() < 1e-5


def test_multivar_adaptive_open_boundary_pipeline():
    device = torch.device("cpu")
    shape = (8, 8, 8)
    pipeline = MultiVarAdaptiveOpenBoundaryPipeline(
        depth=shape[0], height=shape[1], width=shape[2], dx=0.1
    ).to(device)

    q_human = torch.randn(*shape, 4, device=device)
    q_human = q_human / torch.norm(q_human, dim=-1, keepdim=True)

    q_ext = torch.randn(*shape, 4, device=device)
    q_ext = q_ext / torch.norm(q_ext, dim=-1, keepdim=True)

    initial_kappa = pipeline.kappa_boundary

    # Run 5 growth steps
    for _ in range(5):
        m = pipeline.step_multivar_adaptive_growth(q_human, q_ext, dt=0.02)
        assert "delta_phase" in m
        assert "entropy" in m
        assert "kappa_boundary" in m
        assert pipeline.kappa_min <= m["kappa_boundary"] <= pipeline.kappa_max

    assert pipeline.q_sys.shape == (*shape, 4)
    # Norm of q_sys should remain normalized ~ 1
    norm_q = torch.norm(pipeline.q_sys, dim=-1)
    assert torch.allclose(norm_q, torch.ones_like(norm_q), atol=1e-3)


def test_phase_penetration_analyzer():
    shape = (8, 8, 8)
    dx = 0.1
    analyzer = PhasePenetrationAnalyzer(shape, dx=dx)

    dist_map = analyzer.distance_map
    dummy_j_flux = np.exp(-dist_map / 0.25)
    dummy_j_flux = torch.tensor(dummy_j_flux, dtype=torch.float32)

    res = analyzer.analyze(dummy_j_flux)

    assert "delta_phase" in res
    assert "attenuation_coeff" in res
    assert res["delta_phase"] > 0.0
    assert len(res["j_profile"]) == analyzer.max_depth_index + 1


def test_self_human_asymmetry_explorer_integration():
    shape = (8, 8, 8)
    explorer = SelfHumanAsymmetryExplorer(spatial_shape=shape, dx=0.1)

    q_human = torch.randn(*shape, 4)
    q_human = q_human / torch.norm(q_human, dim=-1, keepdim=True)

    q_ext = torch.randn(*shape, 4)
    q_ext = q_ext / torch.norm(q_ext, dim=-1, keepdim=True)

    res = explorer.explore_step(q_human, q_ext, info_context="Test Asymmetry Step")

    assert "step_metrics" in res
    assert "sensor_result" in res
    assert len(explorer.exploration_history) == 1


def test_visualize_meta_topology():
    g_meta = torch.ones(8, 8, 8) * 1.5
    fig = visualize_meta_topology(g_meta, step_info="Test Step", show_plot=False)
    assert fig is not None
