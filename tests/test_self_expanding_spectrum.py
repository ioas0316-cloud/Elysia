r"""
Verification suite for ELYSIA Self-Expanding Spectrum Axis & Causal Mechanism Discovery:
Verifies autonomous discovery of new bipolar axes from unexplained residual variance,
cross-domain expansion (luminance, boundary-tension cloth physics, code/latency),
causal mechanism induction (`query_causal_mechanisms`), arbitrary-resolution slicing,
dynamic Cl(3,0) context rotor projection, and ScaleRotorHierarchy persistence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.embodiment import (
    ScaleRotorHierarchy,
    ScaleSpectralMotorManifold,
    SelfExpandingSpectrumCortex,
    batch_clifford_docking_affinity,
)


def test_discovers_luminance_contrast_axis_from_unexplained_residual_without_hardcoding():
    """Discovers a new dark <-> bright spectrum axis purely from residual SVD when innate axes stay flat."""
    cortex = SelfExpandingSpectrumCortex()
    assert cortex.discovered_axis_count() == 0

    samples = [
        (0.08, 0.92, ["어두움", "그림자"]),
        (0.15, 0.84, ["어두움", "밤"]),
        (0.22, 0.76, ["어두움"]),
        (0.50, 0.50, ["중간빛"]),
        (0.78, 0.20, ["밝음"]),
        (0.86, 0.14, ["밝음", "햇살"]),
        (0.94, 0.06, ["밝음", "빛"]),
    ]
    for lum, shd, tags in samples:
        cortex.observe(
            {
                "warm_cool_ratio": 1.0,
                "motion_flux": 0.14,
                "valence": 0.0,
                "contrast_std": 0.16,
                "luminance": lum,
                "shadow_depth": shd,
            },
            tags=tags,
        )

    assert cortex.discovered_axis_count() == 1
    ax = cortex.get_axis("axis::emergent_1")
    assert ax is not None
    assert "luminance" in ax.weights or "shadow_depth" in ax.weights

    dark_proj = cortex.project({"luminance": 0.05, "shadow_depth": 0.95})["axis::emergent_1"]
    bright_proj = cortex.project({"luminance": 0.96, "shadow_depth": 0.04})["axis::emergent_1"]
    assert abs(bright_proj - dark_proj) > 1.2
    pole_union = set(ax.neg_pole_tags) | set(ax.pos_pole_tags)
    assert "어두움" in pole_union and "밝음" in pole_union


def test_expands_across_avatar_cloth_physics_and_code_domains_with_causal_mechanisms():
    """Autonomously learns new axes and directional causal links across body physics and software execution."""
    cortex = SelfExpandingSpectrumCortex()

    # 1. Domain A: Avatar cloth/muscle trajectory observables along curl phases tau = 0..1
    manifold = ScaleSpectralMotorManifold()
    traj, _ = manifold.capture_or_get_trajectory_rotor([0.60, 0.92, 0.45], [0.08, 0.08, 0.18], steps=36)
    prev_body = None
    for k in range(8):
        fr = traj.sample_at_phase(manifold.chain, k / 7.0, resolution_scale=0.25, tear_threshold=1.12)
        obs = {
            "skin_wrinkle": float(np.max(fr.micro_surface.upper_wrinkle)),
            "skin_sheerness": float(np.max(fr.micro_surface.lower_sheerness)),
            "skin_tear": float(np.max(fr.micro_surface.lower_tear)),
            "elbow_bulge": float(fr.muscles[1].flex_bulge),
        }
        tags = ["근육이완"] if k < 2 else (["주름비침"] if k < 5 else ["고장력찢어짐"])
        cortex.observe(
            obs,
            tags=tags,
            cause_tag="flex_curl" if prev_body is not None else "",
            prev_observables=prev_body,
        )
        prev_body = obs

    # Add opposing relaxation causal transitions
    for k in range(3):
        hi_obs = {"skin_wrinkle": 0.90, "skin_sheerness": 0.85, "skin_tear": 0.55, "elbow_bulge": 1.18}
        lo_obs = {"skin_wrinkle": 0.02 * k, "skin_sheerness": 0.03, "skin_tear": 0.0, "elbow_bulge": 1.01}
        cortex.observe(lo_obs, tags=["근육이완"], cause_tag="relax_release", prev_observables=hi_obs)

    cloth_axis_count = cortex.discovered_axis_count()
    assert cloth_axis_count >= 1
    ax_cloth = cortex.get_axis("axis::emergent_1")
    assert ax_cloth is not None

    # Verify bidirectional causal mechanisms learned along the emergent cloth/muscle axis
    pos_links = cortex.query_causal_mechanisms(ax_cloth.axis_id, direction="+")
    neg_links = cortex.query_causal_mechanisms(ax_cloth.axis_id, direction="-")
    all_causes = {lk.cause_tag for lk in [*pos_links, *neg_links]}
    assert "flex_curl" in all_causes and "relax_release" in all_causes
    if pos_links and neg_links:
        assert {pos_links[0].cause_tag, neg_links[0].cause_tag} == {"flex_curl", "relax_release"}

    # 2. Domain B: Software execution / AST observables -> discovers additional independent axis
    code_obs = [
        (2.0, 4.0, 0.01, "refactor_inline", ["단순화", "빠름"]),
        (3.0, 6.0, 0.02, "refactor_inline", ["명료함"]),
        (6.0, 38.0, 0.08, "add_nested_loop", ["복잡"]),
        (9.0, 120.0, 0.22, "add_nested_loop", ["지연오류"]),
        (11.0, 210.0, 0.35, "add_nested_loop", ["과부하"]),
        (1.5, 3.2, 0.005, "refactor_inline", ["최적화"]),
    ]
    prev_c = {"ast_depth": 5.0, "latency_ms": 25.0, "error_rate": 0.05}
    for depth, lat, err, ctag, tags in code_obs:
        cur_c = {"ast_depth": depth, "latency_ms": lat, "error_rate": err}
        cortex.observe(cur_c, tags=tags, cause_tag=ctag, prev_observables=prev_c)
        prev_c = cur_c

    assert cortex.discovered_axis_count() >= cloth_axis_count + 1


def test_arbitrary_resolution_slicing_and_dynamic_clifford_context_rotor():
    """Supports arbitrary continuous slicing resolution (0.5 vs 0.1) & Cl(3,0) context docking."""
    cortex = SelfExpandingSpectrumCortex()
    for i in range(10):
        v = i / 9.0
        cortex.observe(
            {"luminance": v, "shadow_depth": 1.0 - v},
            tags=["어두움" if v < 0.35 else ("밝음" if v > 0.65 else "중간")],
        )
    ax_id = "axis::emergent_1"
    coarse_bins = cortex.slice_spectrum_at_resolution(ax_id, resolution_step=0.5)
    fine_bins = cortex.slice_spectrum_at_resolution(ax_id, resolution_step=0.1)
    assert len(fine_bins) > len(coarse_bins)

    r_dark_a = cortex.to_context_clifford_rotor({"luminance": 0.05, "shadow_depth": 0.95}, [ax_id])
    r_dark_b = cortex.to_context_clifford_rotor({"luminance": 0.10, "shadow_depth": 0.90}, [ax_id])
    r_bright = cortex.to_context_clifford_rotor({"luminance": 0.95, "shadow_depth": 0.05}, [ax_id])

    affs, _tqs = batch_clifford_docking_affinity(
        np.stack([r_dark_b, r_bright], axis=0),
        r_dark_a,
    )
    assert float(affs[0]) > 0.80
    assert float(affs[1]) < -0.50


def test_unified_hierarchy_axis_registration_and_persistence_roundtrip(tmp_path: Path):
    """Unified graph materializes Scale-3 `axis::*` nodes and persists cortex cleanly across sessions."""
    uni = ScaleRotorHierarchy()
    uni.differentiate_text("빛이 아주 밝다.", spectrum=[0.5, 0.5, 0.3, 0.4])
    uni.differentiate_text("그림자가 어두움이다.", spectrum=[0.5, 0.5, 0.3, -0.2])

    for i in range(8):
        val = i / 7.0
        uni.observe_world_phenomenon_and_expand(
            {"luminance": val, "shadow_depth": 1.0 - val},
            tags=["어두움" if val < 0.4 else "밝음"],
            cause_tag="open_curtain" if i > 0 and val > 0.5 else "",
            prev_observables={"luminance": max(0.0, val - 0.3), "shadow_depth": min(1.0, 1.3 - val)},
        )

    topo = uni.topology_summary()
    assert topo["discovered_axes"] >= 1
    assert "axis::emergent_1" in uni.nodes
    assert uni.nodes["axis::emergent_1"].scale == 3

    save_path = tmp_path / "uni_expanded.json"
    uni.save(save_path)
    reloaded = ScaleRotorHierarchy.load(save_path)
    assert reloaded.topology_summary()["discovered_axes"] == topo["discovered_axes"]
    assert reloaded.spectrum_cortex.project({"luminance": 0.9, "shadow_depth": 0.1}) == uni.spectrum_cortex.project(
        {"luminance": 0.9, "shadow_depth": 0.1}
    )
