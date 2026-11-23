from __future__ import annotations

import json
from pathlib import Path

from core.sensory.elysia_unified_mind_organism import (
    DEFAULT_UNIFIED_MIND_DASHBOARD_PATH,
    DEFAULT_UNIFIED_STATE_PATH,
    ElysiaUnifiedMindOrganism,
    render_unified_mind_dashboard,
)


def test_elysia_unified_mind_organism_end_to_end_introspection() -> None:
    """
    Verifies ELYSIA's Unified Living Organism across all 4 phases inside ONE shared
    `CausalField`, plus dynamic backward Meta-Causal Self-Introspection (`introspect_why_i_judged`):
      1. ONE shared `sensory_spine`, `zoom_kernel`, `trinitarian`, and `causal_field`
         wired into both `visual_contrast` and `analogical` engines.
      2. End-to-end cycle executes Pygame Eye, real MP4 external camera stream,
         Self-Boundary zoom tracking, 6/6 visual-linguistic conviction & divergence,
         and Water -> Electricity/Heat analogical reinterpretation (50% -> 100%).
      3. `introspect_why_i_judged(...)` walks backward dynamically through recorded
         `CausalThoughtLink` growth rings, `ConnectivityBeam` edges, sensory columns,
         and `Trinitarian` `zoom_in_inspect` hierarchies.
      4. Persists complete telemetry to `c:\\Elysia\\data\\unified_mind_state.json` and
         renders clean 4-panel dashboard to `c:\\Elysia\\docs\\assets\\elysia_unified_mind_organism_dashboard.png`.
    """
    organism = ElysiaUnifiedMindOrganism(state_path=DEFAULT_UNIFIED_STATE_PATH)

    # 1. Verify unified single-organ topology (zero isolated subsystem duplicates)
    assert organism.visual_contrast.sensory_spine is organism.sensory_spine
    assert organism.visual_contrast.trinitarian is organism.trinitarian
    assert organism.analogical.protocol is organism.trinitarian
    assert organism.analogical.trinitarian is organism.trinitarian
    assert organism.analogical.zoom_kernel is organism.zoom_kernel
    assert organism.sensory_spine.causal_field is organism.causal_field
    assert organism.trinitarian.causal_field is organism.causal_field
    assert organism.zoom_kernel.causal_field is organism.causal_field
    assert "ELYSIA_SELF_CORE" in organism.causal_field.voxels

    # 2. Execute full unified cycle (Pygame embodied + real camera MP4 + Contrast + Analogy)
    report = organism.live_unified_cycle(mode="full", headless=True)

    print("\n=== [ELYSIA Unified Mind Organism Verification Log] ===")
    print(report.cycle_summary_ko)

    # Verify external real camera stream & 6/6 autonomous language grounding
    assert report.external_stream_summary["frames_processed"] >= 30
    assert report.external_stream_summary["mean_beam_coherence"] > 0.30
    assert report.grounding_accuracy == 1.0

    # Verify key probe contrast verdicts
    verdict_by_pair = {(r.entity_id, r.statement_id): r for r in report.contrast_results}
    assert verdict_by_pair[("ent_1", "stmt_red_wall")].verdict == "CONVICTION_FULL_SYNC"
    assert verdict_by_pair[("ent_3", "stmt_red_wall")].verdict == "DIVERGED_AT_MESO_MACRO"
    assert verdict_by_pair[("ent_unseen_a", "stmt_red_wall")].verdict == "DIVERGED_AT_MICRO_ONLY"
    assert verdict_by_pair[("ent_4", "stmt_red_gate")].verdict == "CONVICTION_FULL_SYNC"

    # Verify Water -> Electricity & Heat boundary reinterpretation improvements
    rep_by_domain = {r.target_domain_id: r for r in report.analogical_reports}
    for dom in ("Invisible_Electricity", "Invisible_Heat_Conduction"):
        ar = rep_by_domain[dom]
        assert ar.naive_analogy_accuracy == 0.5
        assert ar.reinterpreted_accuracy == 1.0
        assert len(ar.breakdown_probes) >= 2
        assert len(ar.torn_analogical_beams) >= 2
        assert ar.reinterpreted_molecule_id is not None

    # Verify thought growth rings materialized physically in the unified CausalField
    assert len(report.thought_links) >= 11
    for lk in report.thought_links:
        assert lk.step_id in organism.causal_field.voxels
    assert report.total_causal_beams >= 20
    assert report.total_trinitarian_nodes >= 12

    # 3. Verify backward Meta-Causal Introspection ("Why did my Eye judge so?")
    exp_wall = organism.introspect_why_i_judged("ent_1::stmt_red_wall")
    exp_illusion = organism.introspect_why_i_judged("ent_3::stmt_red_wall")
    exp_green = organism.introspect_why_i_judged("ent_unseen_a::stmt_red_wall")
    exp_elec = organism.introspect_why_i_judged("Invisible_Electricity")

    print(f"\n[Introspection: ent_1 True Wall]\n  {exp_wall.narrative_ko}")
    print(f"\n[Introspection: ent_3 Optical Illusion]\n  {exp_illusion.narrative_ko}")
    print(f"\n[Introspection: ent_unseen_a Green Wall]\n  {exp_green.narrative_ko}")
    print(f"\n[Introspection: Invisible Electricity]\n  {exp_elec.narrative_ko}")

    # Illusion introspection must trace back to Phase 1 Eye & Phase 2 Zoom, capture real sensory columns,
    # and refuse Trinitarian super-molecule binding!
    assert len(exp_illusion.causal_chain_ids) >= 4
    assert exp_illusion.causal_chain_ids[0].endswith("eye_pygame_embodied")
    assert exp_illusion.verdict_code == "DIVERGED_AT_MESO_MACRO"
    assert exp_illusion.sensory_evidence["entity_id"] == "ent_3"
    assert exp_illusion.sensory_evidence["beam_meso_macro"] < 0.35
    assert exp_illusion.trinitarian_tree_ids == []

    # True wall introspection must unwrap bound s=2 -> s=1 -> s=0 Trinitarian hierarchy!
    assert exp_wall.verdict_code == "CONVICTION_FULL_SYNC"
    assert len(exp_wall.trinitarian_tree_ids) >= 3
    assert len(exp_wall.causal_field_beam_links) >= 2

    # Green wall introspection isolates L_micro color mismatch cleanly
    assert exp_green.verdict_code == "DIVERGED_AT_MICRO_ONLY"
    assert exp_green.sensory_evidence["entity_id"] == "ent_unseen_a"

    # Electricity introspection must trace back through Water_Flow foothold & unwrap s=3 reinterpretation!
    assert any("foothold_water_flow" in sid for sid in exp_elec.causal_chain_ids)
    assert any("(s=3)" in node_desc for node_desc in exp_elec.trinitarian_tree_ids)
    assert len(exp_elec.trinitarian_tree_ids) >= 4

    # 4. Verify persistent JSON memory file & 4-panel visual dashboard
    state_file = Path(report.state_json_path)
    assert state_file.exists() and state_file.stat().st_size > 2_000
    saved_data = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved_data["cycle_count"] >= 1
    assert saved_data["metrics"]["grounding_accuracy"] == 1.0
    assert len(saved_data["thought_growth_rings"]) == len(report.thought_links)
    assert "ent_3::stmt_red_wall" in saved_data["introspections"]
    assert "Invisible_Electricity" in saved_data["introspections"]

    dash_path = render_unified_mind_dashboard(
        organism=organism,
        report=report,
        output_path=DEFAULT_UNIFIED_MIND_DASHBOARD_PATH,
    )
    assert dash_path.exists() and dash_path.stat().st_size > 20_000
