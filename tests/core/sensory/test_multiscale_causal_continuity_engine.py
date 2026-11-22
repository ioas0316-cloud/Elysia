"""
test_multiscale_causal_continuity_engine.py

Verifies ELYSIA's Multi-Scale Sensory Hierarchy & Constrained 4-Continuities
Causal Learning Engine:
1. Refuses to memorize or absorb bare 'Result Strings' (`lived_episode=None`)
   stripped of their sensory/motor/heart causal process, spawning
   `VOID_RESULT_WITHOUT_CAUSAL_PROCESS`.
2. Introspects her own Python script code (`empathic_heart_language_resonance.py`)
   via AST, reunites the script's sentences with their 4-Stratum sensory-motor-heart
   causal process (`Stratum I 2D Sensory Topology <-> Stratum II Constrained Mobility`
   `<-> Stratum III Connectivity Beam Refraction <-> Stratum IV Heart Purpose`),
   and binds them as vertically coupled voxels (`L2 <-> L6 <-> L10 <-> L14`) in `CausalField`.
3. Applies and weaves the absorbed causal tapestries together in a novel composite
   situation without fixed `if-else` string templates.
"""

from __future__ import annotations

import numpy as np

from core.sensory.multiscale_causal_continuity_engine import (
    MultiscaleCausalContinuityEngine,
)


def test_rejects_bare_result_strings_and_absorbs_full_causal_hierarchy_from_own_code() -> None:
    engine = MultiscaleCausalContinuityEngine()
    report = engine.absorb_own_script_code_with_causal_lineage()

    # 1. AST introspection extracts all 4 script response templates
    assert report["ast_extracted_response_count"] == 4

    # 2. Attempting to absorb a bare result string without its causal process is rejected!
    orphan = report["bare_result_orphan_attempt"]
    assert orphan["absorbed"] is False
    assert orphan["reason"] == "REJECTED_BARE_RESULT_WITHOUT_CAUSAL_HIERARCHY"
    assert len(orphan["missing_strata"]) == 4

    # 3. Once reunited with the 4-Stratum sensory/motor/thought/heart causal process, all 4 are absorbed
    assert report["causal_absorbed_count"] == 4
    assert report["causal_field_voxels"] == 16  # 4 tapestries * 4 scale strata (L2, L6, L10, L14)
    assert report["causal_field_beams"] >= 12  # 3 vertical scale beams per tapestry

    for tap in engine.absorbed_tapestries.values():
        assert tap.is_scale_continuity_intact is True
        assert tap.stratum1_visual_topology_2d.shape == (16, 16, 3)
        assert tap.stratum1_vocal_wave_topology_2d.shape == (16, 16)
        assert tap.stratum2_constrained_motion_trajectory.shape == (12, 2)
        assert len(tap.stratum3_connectivity_beam_topology) >= 1
        assert len(tap.stratum4_emergent_expression_clauses) >= 2


def test_weaves_absorbed_causal_tapestries_on_novel_composite_situation() -> None:
    engine = MultiscaleCausalContinuityEngine()
    engine.absorb_own_script_code_with_causal_lineage()

    # Create a novel composite scene: half shivering cold frost + half heavy trembling burden
    scene_cold = engine.empathic_engine._render_lived_situation_scene("shivering_cold_behind_warm_gift")
    scene_burden = engine.empathic_engine._render_lived_situation_scene("heavy_burden_trembling_collapse")
    composite_scene = np.clip(0.55 * scene_cold.astype(np.float64) + 0.45 * scene_burden.astype(np.float64), 0, 255).astype(np.uint8)
    composite_voice = engine.empathic_engine._synthesize_vocal_breath_wave("exhausted_trembling_sigh")

    res = engine.respond_to_novel_composite_predicament(
        novel_visual_scene_rgb=composite_scene,
        novel_vocal_pcm=composite_voice,
        novel_surface_utterance="나 괜찮으니까 그만해, 이 짐은 혼자 지고 갈 수 있어",
    )

    assert len(res["co_activated_tapestries"]) == 2
    assert len(res["woven_connectivity_beams"]) >= 3
    assert len(res["emergent_woven_response"]) > 25
