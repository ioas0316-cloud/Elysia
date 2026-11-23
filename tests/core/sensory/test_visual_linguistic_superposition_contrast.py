from __future__ import annotations

from pathlib import Path

import numpy as np

from core.sensory.visual_linguistic_superposition_contrast_engine import (
    DEFAULT_CONTRAST_DASHBOARD_PATH,
    VisualLinguisticSuperpositionEngine,
    build_korean_statement_catalog,
    render_visual_linguistic_contrast_dashboard,
)


def test_visual_linguistic_superposition_and_layer_divergence() -> None:
    """
    Verifies ELYSIA's Visual-Linguistic Superposition & Contrast Principle:
      1. True match (`ent_1` vs "빨간 벽이 길을 막는다") -> full L_micro/meso/macro conviction + grounding.
      2. Deceptive optical illusion (`ent_3` vs "빨간 벽이 길을 막는다") -> agrees at L_micro,
         explicitly detects divergence at L_meso & L_macro, and refuses binding!
      3. Appearance mismatch / structural law match (`ent_unseen_a` green wall vs "빨간 벽") ->
         explicitly identifies divergence at L_micro ONLY while matching L_meso & L_macro!
      4. Autonomous best-statement selection across all 6 entities yields 6/6 exact grounding.
    """
    engine = VisualLinguisticSuperpositionEngine()
    engine.observe_visual_world(headless=True)
    catalog = build_korean_statement_catalog()

    # Key 4 contrast cases probing agreement vs divergence loci
    c_true_wall = engine.superpose_and_contrast("ent_1", catalog["stmt_red_wall"])
    c_deceptive_illusion = engine.superpose_and_contrast("ent_3", catalog["stmt_red_wall"])
    c_green_vs_red_wall = engine.superpose_and_contrast("ent_unseen_a", catalog["stmt_red_wall"])
    c_true_gate = engine.superpose_and_contrast("ent_4", catalog["stmt_red_gate"])

    key_contrasts = [c_true_wall, c_deceptive_illusion, c_green_vs_red_wall, c_true_gate]

    print("\n=== [ELYSIA Visual-Linguistic Superposition & Contrast Log] ===")
    for res in key_contrasts:
        print(
            f"\n[{res.entity_id} vs {res.statement_id}] verdict={res.verdict} "
            f"(sync: micro={res.sync_micro:.2f}, meso={res.sync_meso:.2f}, macro={res.sync_macro:.2f})"
        )
        print(f"  matched={res.matched_layers} | diverged={res.diverged_layers} | grounded={res.grounded_super_molecule_id}")
        print(f"  -> {res.explanation_ko}")

    # Full 6x6 entity x statement conviction matrix & autonomous best selection
    entity_ids = ["ent_1", "ent_2", "ent_3", "ent_4", "ent_unseen_a", "ent_unseen_b"]
    stmt_ids = [
        "stmt_red_wall",
        "stmt_red_crate",
        "stmt_red_illusion",
        "stmt_red_gate",
        "stmt_green_wall",
        "stmt_green_illusion",
    ]
    conv_mat = np.zeros((len(entity_ids), len(stmt_ids)), dtype=np.float64)
    for ri, eid in enumerate(entity_ids):
        for ci, sid in enumerate(stmt_ids):
            r = engine.superpose_and_contrast(eid, catalog[sid], bind_on_conviction=False)
            conv_mat[ri, ci] = r.overall_conviction

    best_matches = {
        eid: engine.ground_entity_to_best_statement(eid, catalog) for eid in entity_ids
    }
    print("\n[Autonomous Multi-Scale Statement Grounding across all 6 Entities]")
    for eid, bres in best_matches.items():
        print(f"  {eid:14s} -> {bres.statement_id:20s} | conviction={bres.overall_conviction:.3f} | {bres.text_ko}")

    dash_path = render_visual_linguistic_contrast_dashboard(
        engine=engine,
        key_contrasts=key_contrasts,
        conviction_matrix=conv_mat,
        entity_ids=entity_ids,
        stmt_ids=stmt_ids,
        output_path=Path(DEFAULT_CONTRAST_DASHBOARD_PATH),
    )
    assert dash_path.exists() and dash_path.stat().st_size > 10_000

    # --- Case 1: Full conviction on true red wall ---
    assert c_true_wall.verdict == "CONVICTION_FULL_SYNC"
    assert c_true_wall.sync_micro > 0.90
    assert c_true_wall.sync_meso > 0.85
    assert c_true_wall.sync_macro > 0.85
    assert c_true_wall.grounded_super_molecule_id is not None

    # --- Case 2: Illusion catches Meso/Macro lie despite identical L_micro color ---
    assert c_deceptive_illusion.verdict == "DIVERGED_AT_MESO_MACRO"
    assert c_deceptive_illusion.sync_micro > 0.90
    assert c_deceptive_illusion.sync_meso < 0.35
    assert "L_micro" in c_deceptive_illusion.matched_layers
    assert "L_meso" in c_deceptive_illusion.diverged_layers
    assert c_deceptive_illusion.grounded_super_molecule_id is None

    # --- Case 3: Green wall catches L_micro color diff while preserving L_meso/macro invariant ---
    assert c_green_vs_red_wall.verdict == "DIVERGED_AT_MICRO_ONLY"
    assert c_green_vs_red_wall.sync_micro < 0.65
    assert c_green_vs_red_wall.sync_meso > 0.85
    assert c_green_vs_red_wall.sync_macro > 0.85
    assert "L_micro" in c_green_vs_red_wall.diverged_layers
    assert "L_meso" in c_green_vs_red_wall.matched_layers

    # --- Case 4: Gate emergent transition boundary conviction ---
    assert c_true_gate.verdict == "CONVICTION_FULL_SYNC"
    assert c_true_gate.sync_macro > 0.85

    # --- Case 5: Exact 6/6 autonomous grounded selection ---
    expected_map = {
        "ent_1": "stmt_red_wall",
        "ent_2": "stmt_red_crate",
        "ent_3": "stmt_red_illusion",
        "ent_4": "stmt_red_gate",
        "ent_unseen_a": "stmt_green_wall",
        "ent_unseen_b": "stmt_green_illusion",
    }
    for eid, exp_sid in expected_map.items():
        assert best_matches[eid].statement_id == exp_sid
        assert best_matches[eid].verdict == "CONVICTION_FULL_SYNC"
        assert best_matches[eid].grounded_super_molecule_id is not None
