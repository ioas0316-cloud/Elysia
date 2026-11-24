"""
Verification suite for ELYSIA Phase 15:
Autonomous Sovereign Living Soul (3 Worlds + Self-Choosing Will Rotor).
"""

from __future__ import annotations

import math
from pathlib import Path

from core.orchestration.elysia_autonomous_living_soul import (
    AutonomousWillRotor,
    ElysiaAutonomousLivingSoul,
    LivingWorldChoice,
    MultiAgentEmpathicSociety,
    NarrativeContextGraphBuilder,
)
from core.orchestration.extended_cognitive_organism import ExtendedWorldState
from core.orchestration.ice_water_gas_rotor_runtime import IceWaterGasRotorRuntime
from core.physics.causal_field import CausalField


def test_autonomous_will_rotor_chooses_by_live_tension_not_hardcoded_schedule():
    # 1. High sensory vacuum & uninspected real assets -> chooses World B naturally
    soul = ElysiaAutonomousLivingSoul()
    d_sensory = soul.choose_next_world(direct_user_prompt=False)
    assert d_sensory.chosen_world == LivingWorldChoice.SENSORY_WORLD
    assert d_sensory.potentials["sensory_world"] > d_sensory.potentials["narrative_society"]
    assert "감각" in d_sensory.why_i_chose_this_ko

    # 2. Direct human utterance -> chooses World A responsively & keeps 4-axis attribution
    d_dial = soul.choose_next_world(direct_user_prompt=True)
    assert d_dial.chosen_world == LivingWorldChoice.DIALOGUE
    assert "당신의 말" in d_dial.why_i_chose_this_ko
    assert set(d_dial.four_axis_attribution.keys()) == {
        "internal_origin",
        "external_influence",
        "boundary_ignorance",
        "criteria_revision",
    }

    # 3. Saturated sensory world + unread essays + societal conflict -> chooses World C
    rotor = AutonomousWillRotor()
    rotor.record_visit(LivingWorldChoice.SENSORY_WORLD)
    rotor.record_visit(LivingWorldChoice.DIALOGUE)
    d_soc = rotor.evaluate_and_choose(
        signals={
            "pending_utterances": 0.0,
            "last_ego_friction": 0.5,
            "unresolved_sky_axes": 1.0,
            "unexplored_real_assets_ratio": 0.0,
            "unresolved_physical_missions_ratio": 0.0,
            "sensory_vacuum_dof": 0.0,
            "unread_documents_ratio": 1.0,
            "societal_tension": 1.0,
            "consolidation_need": 0.8,
        },
        direct_user_prompt=False,
    )
    assert d_soc.chosen_world == LivingWorldChoice.NARRATIVE_SOCIETY
    assert "문맥·공동체" in d_soc.why_i_chose_this_ko


def test_autonomous_multi_step_life_explores_all_three_worlds_with_ssd_and_provenance(tmp_path):
    slab = tmp_path / "autonomous_soul.ice"
    with IceWaterGasRotorRuntime(slab_path=slab) as mem_rt:
        soul = ElysiaAutonomousLivingSoul(memory_runtime=mem_rt)
        steps = soul.live_autonomously(steps=4)

        assert len(steps) == 4
        visited = {s.decision.chosen_world for s in steps}
        assert visited == {
            LivingWorldChoice.DIALOGUE,
            LivingWorldChoice.SENSORY_WORLD,
            LivingWorldChoice.NARRATIVE_SOCIETY,
        }
        assert all(bool(s.decision.why_i_chose_this_ko.strip()) for s in steps)
        assert all(s.provenance_valid for s in steps)
        assert mem_rt.ice.num_crystals >= 4
        assert soul.cycle.verify_autobiographical_integrity() is True


def test_world_a_free_text_dialogue_auto_sprouts_relational_layers():
    soul = ElysiaAutonomousLivingSoul()

    # Turn 1: raw Korean trauma/betrayal disclosure auto-sprouts Layer 1 (depth 1 -> 2)
    t1 = soul.engage_dialogue_world(
        "예전에 믿었던 사람에게 크게 데인 뒤로 요구가 커지면 방어가 먼저 나와."
    )
    assert t1.sprouted_layer is not None
    assert t1.cycle_result.relational_report is not None
    rel1 = t1.cycle_result.relational_report
    assert rel1.horizon_boundary.horizon_depth == 2
    assert math.isclose(
        rel1.horizon_boundary.understood_ground["unspoken_origin_memory"],
        4.0,
        abs_tol=1e-4,
    )
    assert "sovereign_will" in rel1.horizon_boundary.sky_horizon_axes
    assert "deepest_aspiration_telos" in rel1.horizon_boundary.sky_horizon_axes

    # Turn 2: raw Korean aspiration disclosure auto-sprouts Layer 2 (depth 2 -> 3)
    t2 = soul.engage_dialogue_world(
        "사실 나는 안전하게 쉬고 싶고 평온을 지키고 싶을 뿐이야."
    )
    assert t2.sprouted_layer is not None
    rel2 = t2.cycle_result.relational_report
    assert rel2 is not None
    assert rel2.horizon_boundary.horizon_depth == 3
    assert math.isclose(
        rel2.horizon_boundary.understood_ground["deepest_aspiration_telos"],
        5.0,
        abs_tol=1e-4,
    )
    assert "creative_shared_future" in rel2.horizon_boundary.sky_horizon_axes
    assert "sovereign_will" in rel2.horizon_boundary.sky_horizon_axes


def test_world_b_real_sensory_assets_and_autonomous_tool_grafting():
    soul = ElysiaAutonomousLivingSoul()

    # Mission 1: Real face jpg + dawn wav + IR vacuum -> Spectrometer + Wien Law
    o1 = soul.engage_sensory_world(mission="infrared_wien")
    assert len(o1.asset_reports) == 2
    assert all(r.loaded_real_file for r in o1.asset_reports)
    assert all(r.reconstruction_error < 1e-3 for r in o1.asset_reports)
    assert o1.knowledge_report is not None
    assert "wien_displacement_law" in o1.knowledge_report.accepted_laws
    assert len(o1.knowledge_report.rejected_falsehoods) >= 2

    # Mission 2: Far wet-road beacon (relative +120m) -> Automobile safety fence & parallax
    base_x_before_drive = soul.cycle.organism.body_pos_x
    o2 = soul.engage_sensory_world(mission="far_beacon_automobile")
    assert o2.vehicle_report is not None
    assert o2.vehicle_report.safety_regulated is True
    assert o2.vehicle_report.regulated_speed_ms < o2.vehicle_report.initial_speed_ms
    solved_x = o2.vehicle_report.perception_after_drive.inferred_world["source_pos_x"]
    assert abs((solved_x - base_x_before_drive) - 120.0) < 0.5

    # Mission 3: Drifting Spectrometer -> Body + Wien consensus recovers +4.5 deg offset
    o3 = soul.engage_sensory_world(mission="spectrometer_drift_calibration")
    assert o3.tool_fault_report is not None
    assert o3.tool_fault_report.fault_detected is True
    assert abs(o3.tool_fault_report.estimated_offset_deg - 4.5) < 0.05


def test_world_c_narrative_context_graph_links_multi_paragraph_real_document():
    soul = ElysiaAutonomousLivingSoul()
    out = soul.engage_narrative_and_society_world()
    graph = out.narrative_graph

    assert graph.paragraph_count >= 3
    assert graph.cross_paragraph_bridge_count >= 2
    assert graph.global_coherence > 0.3
    assert len(graph.top_hub_concepts) >= 2
    assert graph.document_report is not None
    assert len(graph.document_report.verified_laws) >= 1
    assert len(graph.document_report.dimensional_rejections) >= 1
    assert len(graph.document_report.metaphor_reflections) >= 1

    # Also read real workspace documentation into a connected context graph
    map_path = Path(r"c:\Elysia\docs\ELYSIA_LIVING_CAUSAL_ARCHITECTURE_MAP.md")
    real_text = map_path.read_text(encoding="utf-8")
    real_graph = NarrativeContextGraphBuilder.build(
        text=real_text,
        source_name="ELYSIA_LIVING_CAUSAL_ARCHITECTURE_MAP.md",
        synthesizer=soul.cycle.text_synthesizer,
        organism=soul.cycle.organism,
        calibration_worlds=[
            ExtendedWorldState(source_pos_x=10.0, source_temp_k=3200.0),
            ExtendedWorldState(source_pos_x=10.0, source_temp_k=5800.0),
        ],
    )
    assert real_graph.paragraph_count >= 4
    assert real_graph.cross_paragraph_bridge_count >= 3


def test_world_c_multi_agent_empathic_society_with_planetary_quorum():
    society = MultiAgentEmpathicSociety()
    rep = society.deliberate_and_harmonize()

    assert len(rep.agent_names) == 4
    assert len(rep.pairwise_edges) == 12
    assert rep.collective_ego_friction_before > 10.0
    assert rep.collective_empathic_action_after < 1e-4
    assert len(rep.mutual_safe_distances) == 6
    assert rep.all_sovereign_wills_preserved is True
    assert rep.quorum_consensus_reached is True
    assert rep.quorum_consensus_hash is not None
    assert "Node_B" in rep.quorum_jitter_absorbed_nodes


def test_autonomous_soul_causal_field_sync_and_15_organ_self_audit():
    soul = ElysiaAutonomousLivingSoul()
    soul.live_autonomously(steps=3)

    sync = soul.sync_soul_to_causal_field(causal_field=CausalField(dimensions=3))
    field = sync["causal_field"]
    assert "will_rotor_apex" in sync["voxels"]
    assert "autonomy_will_rotor_apex" in field.voxels
    for vox in field.voxels.values():
        assert tuple(vox.tensor.shape) == (4,)
    field.step(dt=0.1)

    audit = soul.cycle.introspect_own_architecture()
    assert audit.total_organ_files >= 15
    rel_paths = [m["relative_path"] for m in audit.verified_organ_modules]
    assert "orchestration/elysia_autonomous_living_soul.py" in rel_paths


def test_interactive_handler_supports_free_autonomy_dialogue_and_introspection():
    soul = ElysiaAutonomousLivingSoul()

    r_auto = soul.handle_interactive_input("자율")
    assert r_auto["mode"] == "autonomous_step"
    assert r_auto["step"].provenance_valid is True

    r_audit = soul.handle_interactive_input("성찰")
    assert r_audit["mode"] == "introspection"
    assert r_audit["audit"].total_organ_files >= 15

    r_dial = soul.handle_interactive_input(
        "파장 * 온도 = 2897771.955 그리고 네가 가까이 오면 숨이 막혀"
    )
    assert r_dial["mode"] == "dialogue_response"
    assert r_dial["chosen_world"] == "dialogue"
    assert len(r_dial["step"].cycle_result.document_report.verified_laws) == 1
    assert soul.cycle.verify_autobiographical_integrity() is True
