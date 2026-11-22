r"""
test_recursive_fractal_meta_rotor_engine.py

Empirical verification of ELYSIA's 2nd-Order Recursive Fractal Meta-Rotor:
  1. Loads her own crystallized 4-world engrams (`universal_4world_engrams.json`).
  2. Swallows 5 of her own live Python source files (`causal_field.py`,
     `ocular_scale_organ.py`, `spacetime_spectral_boundary_core.py`,
     `agamotto_causal_rotor_engram_engine.py`, `elysia_live_organism_runtime.py`).
  3. Swallows all 2,529 records of `engram_index.json` as ONE historical archive rotor.
  4. Lifts ALL 10 engrams recursively into 2nd-order Meta-Rotor Crystals, verifies
     >90% explained variance, <0.5ms child reconstruction, code bifurcation line discovery,
     and unseen future trajectory extrapolation from only the early half (`theta <= 0.5`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.sensory.recursive_fractal_meta_rotor_engine import (
    RecursiveFractalMetaRotorEngine,
)

UNIVERSAL_ENGRAM_PATH = Path(r"c:\Elysia\data\universal_4world_engrams.json")
ENGRAM_INDEX_PATH = Path(r"c:\Elysia\data\topology\engram_index.json")
META_DASHBOARD_OUT = Path(
    r"c:\Elysia\docs\assets\elysia_recursive_fractal_meta_rotor_dashboard.png"
)

SELF_CODEBASE_FILES = [
    Path(r"c:\Elysia\L1_causal_field\core\physics\causal_field.py"),
    Path(r"c:\Elysia\L1_causal_field\core\sensory\ocular_scale_organ.py"),
    Path(r"c:\Elysia\L1_causal_field\core\sensory\spacetime_spectral_boundary_core.py"),
    Path(r"c:\Elysia\L1_causal_field\core\sensory\agamotto_causal_rotor_engram_engine.py"),
    Path(r"c:\Elysia\L1_causal_field\core\sensory\elysia_live_organism_runtime.py"),
]


def test_recursive_meta_rotor_swallows_4worlds_own_codebase_and_2529_archive() -> None:
    meta_eng = RecursiveFractalMetaRotorEngine()

    # 1. Swallow her own crystallized 4-world engram file
    assert UNIVERSAL_ENGRAM_PATH.exists(), f"Missing {UNIVERSAL_ENGRAM_PATH}"
    n_worlds = meta_eng.rotor_engine.load_engrams(UNIVERSAL_ENGRAM_PATH)
    assert n_worlds == 4
    world_ids = list(meta_eng.rotor_engine.engrams.keys())

    meta_4world = meta_eng.crystallize_engrams_into_meta_rotor(
        meta_id="meta_4world_trunk", engram_ids=world_ids, max_rank=3
    )
    print(
        f"\n[meta_4world_trunk] var={meta_4world.explained_variance_ratio:.4f}, "
        f"recon_cos={meta_4world.mean_reconstruction_cosine:.4f}, "
        f"time={meta_4world.crystallization_ms:.3f}ms"
    )
    assert meta_4world.explained_variance_ratio > 0.90
    assert meta_4world.mean_reconstruction_cosine > 0.97

    mp4_recon, recon_ms = meta_eng.reconstruct_child_from_meta_rotor(
        "meta_4world_trunk", "real_mp4_flower"
    )
    mp4_orig = np.asarray(
        meta_eng.rotor_engine.engrams["real_mp4_flower"].wavelength_knots_8d,
        dtype=np.float32,
    )
    assert recon_ms < 0.5
    assert meta_eng._cosine(mp4_orig, mp4_recon) > 0.97

    # 2. Swallow 5 of ELYSIA's own living Python source files
    code_ids = []
    hotspots = []
    for py_p in SELF_CODEBASE_FILES:
        c_eng, hotspot = meta_eng.swallow_python_source_file_as_rotor(py_p)
        code_ids.append(c_eng.engram_id)
        hotspots.append(hotspot)
        assert 0 < hotspot.peak_bifurcation_line < hotspot.total_lines
        print(
            f"[Swallowed Self Code] {hotspot.file_name:40s} | lines={hotspot.total_lines:4d} "
            f"| peak_bifurcation=L{hotspot.peak_bifurcation_line} | depth={hotspot.mean_nesting_depth:.2f}"
        )

    meta_codebase = meta_eng.crystallize_engrams_into_meta_rotor(
        meta_id="meta_elysia_self_codebase", engram_ids=code_ids, max_rank=3
    )
    print(
        f"[meta_elysia_self_codebase] var={meta_codebase.explained_variance_ratio:.4f}, "
        f"recon_cos={meta_codebase.mean_reconstruction_cosine:.4f}"
    )
    assert meta_codebase.mean_reconstruction_cosine > 0.95

    # 3. Swallow all 2,529 records from engram_index.json
    arch_eng, rec_count = meta_eng.swallow_archive_engram_index_as_rotor(ENGRAM_INDEX_PATH)
    assert rec_count == 2529
    print(f"[Swallowed Historical Archive] {rec_count} records -> 1 rotor ({arch_eng.first_pass_crystallization_ms:.2f}ms)")

    # 4. Lift ALL 10 engrams together into the 2nd-Order Grand World-Tree Meta-Rotor
    all_10_ids = world_ids + code_ids + [arch_eng.engram_id]
    meta_grand = meta_eng.crystallize_engrams_into_meta_rotor(
        meta_id="meta_grand_world_tree", engram_ids=all_10_ids, max_rank=4
    )
    print(
        f"[meta_grand_world_tree] N={len(all_10_ids)}, var={meta_grand.explained_variance_ratio:.4f}, "
        f"recon_cos={meta_grand.mean_reconstruction_cosine:.4f}, time={meta_grand.crystallization_ms:.3f}ms"
    )
    assert meta_grand.explained_variance_ratio > 0.90
    assert meta_grand.mean_reconstruction_cosine > 0.95
    assert "meta_grand_world_tree" in meta_eng.causal_field.voxels
    assert len(meta_eng.causal_field.beams) >= 10

    # 5. Predict unseen future knots (theta > 0.5) purely from slightly perturbed early knots (theta <= 0.5)
    rng = np.random.default_rng(7)
    early_obs = np.maximum(mp4_orig[:5] + rng.normal(0.0, 0.005, size=mp4_orig[:5].shape), 0.0)
    analogy = meta_eng.predict_unseen_future_via_meta_rotor(
        "meta_4world_trunk", early_obs, mp4_orig, query_id="perturbed_early_flower"
    )
    print(
        f"[Meta-Rotor Future Extrapolation] nearest={analogy.nearest_world_id} "
        f"(sim={analogy.similarity:.4f}) | future_cos={analogy.predicted_future_cosine:.4f} "
        f"| {analogy.latency_ms:.4f}ms"
    )
    assert analogy.nearest_world_id == "real_mp4_flower"
    assert analogy.predicted_future_cosine > 0.90
    assert analogy.latency_ms < 1.0

    # 6. Render visual dashboard PNG
    out_png = meta_eng.render_fractal_meta_rotor_dashboard(
        meta_4world=meta_4world,
        meta_codebase=meta_codebase,
        meta_grand=meta_grand,
        code_hotspots=hotspots,
        analogy_res=analogy,
        mp4_orig_theta1=mp4_orig[-1],
        mp4_recon_theta1=mp4_recon[-1],
        output_path=META_DASHBOARD_OUT,
    )
    assert out_png.exists() and out_png.stat().st_size > 25000
