"""
Verification suite for ELYSIA v3.1:
Hierarchical Open Scale-Rotor Interactive Causal Architecture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.orchestration.elysia_unified_living_cycle import ElysiaUnifiedLivingCycle
from core.orchestration.elysia_v31_open_scale_rotor import (
    CausalDroplet,
    ElysiaV31OpenScaleRotorSystem,
    FoggyRoadMultiFieldEnvironment,
    HierarchyRelation,
    InteractionProtocol,
    OpaqueSensoryMedium,
    OpenVirtualRelationalMemory,
    RealExternalToolBridge,
    UnlabeledSpatialObjectField,
    derive_structural_principle_label_free,
)
from core.orchestration.ice_water_gas_rotor_runtime import IceWaterGasRotorRuntime
from core.physics.causal_field import CausalField


def test_scenario1_spontaneous_spectrum_correlation_axis_without_prelabels():
    """Scenario 1 (Chili-Spiciness principle): Zero pre-labels, pure spectrum correlation."""
    memory = OpenVirtualRelationalMemory()
    protocol = InteractionProtocol(memory)
    medium = OpaqueSensoryMedium(seed=11)

    res = protocol.interact_and_discover_axis(
        medium_name="unlabeled_chemical_medium",
        interact_fn=medium.interact,
        stimulus_channel="substance_dose",
        sweep_values=[0.0, 0.5, 1.5, 3.0, 5.0, 2.0, 4.0],
        scale_tier="earth",
        scale_radius=1.1,
    )

    axis = res["axis"]
    assert axis is not None
    assert res["responsive_channels"] == ["receptor_ch3"]
    assert abs(axis.dominant_correlation) > 0.90
    assert axis.monotonicity > 0.95
    assert set(res["invariant_channels"]) == {
        "receptor_ch0",
        "receptor_ch1",
        "receptor_ch2",
        "receptor_ch4",
    }
    assert axis.axis_id in memory.axes
    assert axis.axis_id in memory.nodes


def test_fog_modality_self_assessment_switches_field_without_hardcoded_fog_rule():
    """Car-in-Fog principle: Empirical modality self-assessment without hardcoded fog rules."""
    # Clear air: both visible optics and EM radar preserve clear distance correlation
    mem_clear = OpenVirtualRelationalMemory()
    proto_clear = InteractionProtocol(mem_clear)
    clear_env = FoggyRoadMultiFieldEnvironment(fog_density=0.0, seed=23)
    clear_report = proto_clear.assess_and_select_viable_modality(
        modalities={
            "visible_optical_field": clear_env.probe_optical,
            "em_radar_field": clear_env.probe_em_radar,
        },
        stimulus_channel="obstacle_distance_m",
        sweep_values=[5.0, 12.0, 20.0, 35.0, 50.0, 70.0],
    )
    assert "visible_optical_field" in clear_report.viable_modalities
    assert "em_radar_field" in clear_report.viable_modalities

    # Dense fog: visible optical resolution collapses -> autonomously selects EM radar
    mem_fog = OpenVirtualRelationalMemory()
    proto_fog = InteractionProtocol(mem_fog)
    fog_env = FoggyRoadMultiFieldEnvironment(fog_density=0.92, seed=23)
    fog_report = proto_fog.assess_and_select_viable_modality(
        modalities={
            "visible_optical_field": fog_env.probe_optical,
            "em_radar_field": fog_env.probe_em_radar,
        },
        stimulus_channel="obstacle_distance_m",
        sweep_values=[5.0, 12.0, 20.0, 35.0, 50.0, 70.0],
    )
    assert "visible_optical_field" in fog_report.unusable_modalities
    assert fog_report.modality_scores["visible_optical_field"]["sensitivity_resolution"] < 0.25
    assert fog_report.selected_modality == "em_radar_field"
    assert fog_report.modality_scores["em_radar_field"]["max_abs_correlation"] > 0.95
    assert fog_report.formed_axis is not None
    assert any(ev.friction_kind == "SensoryResolutionCollapse" for ev in mem_fog.boundary_events)


def test_scenario2a_apple_principle_label_free_structural_causalization():
    """Scenario 2A (Apple principle): Structural principle emerged strictly from probing."""
    memory = OpenVirtualRelationalMemory()
    protocol = InteractionProtocol(memory)
    obj_field = UnlabeledSpatialObjectField(radius_cm=4.2, stiffness=6.5)

    res = derive_structural_principle_label_free(protocol, obj_field)

    assert res["closed_convex_contour"] is True
    assert res["long_to_short_wave_ratio"] > 2.5
    assert res["firm_surface_tension"] is True
    assert res["tactile_axis"] is not None
    assert res["droplet"].droplet_id in memory.droplets
    assert res["droplet"].droplet_id in memory.nodes


def test_scenario2b_atom_molecule_self_structure_hierarchy_discernment_and_bonding():
    """Scenario 2B (Atom-Molecule-Macro): Self-structure discernment with zero external tier tags."""
    memory = OpenVirtualRelationalMemory()

    a1 = CausalDroplet("unit_alpha_1", scale_radius=0.80, surface_tension=0.45, spectral_signature=np.array([1.0, 0.2, 0.1, 0, 0, 0, 0, 0.2]))
    a2 = CausalDroplet("unit_alpha_2", scale_radius=0.85, surface_tension=0.48, spectral_signature=np.array([0.8, 0.4, 0.1, 0.1, 0, 0, 0, 0.1]))
    a3 = CausalDroplet("unit_alpha_3", scale_radius=0.82, surface_tension=0.46, spectral_signature=np.array([0.9, 0.3, 0.2, 0, 0, 0, 0, 0.1]))
    a4 = CausalDroplet("unit_alpha_4", scale_radius=0.88, surface_tension=0.47, spectral_signature=np.array([0.7, 0.5, 0.2, 0.1, 0, 0, 0, 0.1]))
    a5 = CausalDroplet("unit_alpha_5", scale_radius=0.90, surface_tension=0.49, spectral_signature=np.array([0.6, 0.4, 0.3, 0.2, 0, 0, 0, 0.1]))

    m1 = a1.bond_with(a2, "composite_m1")
    m2 = a3.bond_with(a4, "composite_m2")
    macro_pair = m1.bond_with(m2, "macro_cluster_0")
    g1 = macro_pair.bond_with(a5, "macro_enclosure_g1")

    assert a1.discern_against_self(a2).relation == HierarchyRelation.PEER_ELEMENTARY
    assert m1.discern_against_self(a3).relation == HierarchyRelation.SUB_TIER_CONSTITUENT
    assert m1.discern_against_self(m2).relation == HierarchyRelation.PEER_TIER_COMPOSITE
    assert m1.discern_against_self(g1).relation == HierarchyRelation.SUPER_TIER_ENCLOSURE

    for dr in (a1, a2, a3, a4, a5, m1, m2, macro_pair, g1):
        memory.absorb_droplet(dr)

    metrics = memory.topology_metrics()
    assert set(metrics["scale_tiers_occupied"]) == {"earth", "water", "sky"}
    assert metrics["warp_links"] > 0
    assert metrics["weft_links"] > 0


def test_spherical_virtual_relational_memory_warp_weft_and_missing_node_inference():
    """Spherical mirror geodesic inference reconstructs an occluded node from live neighbors."""
    memory = OpenVirtualRelationalMemory()
    a1 = CausalDroplet("u1", scale_radius=0.80, surface_tension=0.50, spectral_signature=np.array([1.0, 0.3, 0.1, 0, 0, 0, 0, 0]))
    a2 = CausalDroplet("u2", scale_radius=0.82, surface_tension=0.52, spectral_signature=np.array([0.9, 0.4, 0.1, 0, 0, 0, 0, 0]))
    a3 = CausalDroplet("u3", scale_radius=0.84, surface_tension=0.51, spectral_signature=np.array([0.85, 0.35, 0.15, 0, 0, 0, 0, 0]))
    m1 = a1.bond_with(a2, "m1")
    m2 = a2.bond_with(a3, "m2")
    macro = m1.bond_with(m2, "macro")

    for dr in (a1, a2, a3, m1, m2, macro):
        memory.absorb_droplet(dr)

    # Occlude m1 while its warp & weft spherical neighbors stay observable around 0.75
    live_values = {
        "u1": 0.72,
        "u2": 0.78,
        "m2": 0.76,
        "macro": 0.74,
    }
    inferred = memory.infer_missing_node_value("m1", live_values)
    assert inferred["inferred"] is True
    assert inferred["confidence"] > 0.50
    assert abs(inferred["estimated_value"] - 0.75) < 0.08
    assert len(inferred["neighbors_used"]) >= 3

    unknown = memory.infer_missing_node_value("nonexistent_node", live_values)
    assert unknown["inferred"] is False
    assert unknown["estimated_value"] is None


def test_scenario3_real_external_subprocess_filesystem_media_and_non_crashing_error_absorption():
    """Scenario 3: Genuine external OS subprocesses, workspace ASTs, real media & fault absorption."""
    memory = OpenVirtualRelationalMemory()
    protocol = InteractionProtocol(memory)
    bridge = RealExternalToolBridge(Path(r"c:\Elysia"))

    before = memory.topology_metrics()
    sub_res = bridge.probe_real_python_subprocess(protocol)
    ws_res = bridge.probe_real_workspace_files_and_media(protocol)
    after = memory.topology_metrics()

    assert sub_res["compute_axis"] is not None
    assert sub_res["compute_axis"].dominant_correlation > 0.85
    assert len(sub_res["fault_boundary_events"]) == 1
    assert "ZeroDivisionError" in sub_res["fault_boundary_events"][0].detail

    assert ws_res["files_scanned"] >= 3
    assert ws_res["code_axis"] is not None
    assert len(ws_res["media_droplets"]) == 2
    media_ids = {d.droplet_id for d in ws_res["media_droplets"]}
    assert "external_media::real_human_face_hd" in media_ids
    assert "external_media::unified_creation_dawn" in media_ids

    assert after["node_count"] > before["node_count"]
    assert (after["warp_links"] + after["weft_links"]) > (before["warp_links"] + before["weft_links"])


def test_v31_scale_rotor_causal_field_4d_sync_ssd_and_19_organ_audit(tmp_path: Path):
    """Full v3.1 orchestration with SSD Ice-Water-Gas persistence, 4D CausalField sync, and 19-organ self-audit."""
    slab_path = tmp_path / "elysia_v31.ice"
    with IceWaterGasRotorRuntime(slab_path=slab_path) as mem_rt:
        cycle = ElysiaUnifiedLivingCycle(memory_runtime=mem_rt)
        system = ElysiaV31OpenScaleRotorSystem(workspace_root=Path(r"c:\Elysia"), living_cycle=cycle)

        report = system.run_all_v31_scenarios()
        assert report["final_topology"]["node_count"] >= 14
        assert report["final_topology"]["warp_links"] > 0
        assert report["final_topology"]["weft_links"] > 0
        assert mem_rt.ice.num_crystals >= 3
        assert cycle.verify_autobiographical_integrity() is True

        field = CausalField(dimensions=3)
        sync = system.memory.sync_to_causal_field(causal_field=field)
        assert sync["beams_linked"] > 0
        assert len(field.voxels) == len(system.memory.nodes)
        for vox in field.voxels.values():
            assert tuple(vox.tensor.shape) == (4,)
        field.step(dt=0.1)

        audit = cycle.introspect_own_architecture()
        assert audit.total_organ_files >= 19
        assert "orchestration/elysia_v31_open_scale_rotor.py" in [
            m["relative_path"] for m in audit.verified_organ_modules
        ]


def test_guardrails_zero_crash_on_malformed_or_contradictory_external_stimuli():
    """Guardrail §5: Unexpected external errors or malformed payloads NEVER force termination."""
    memory = OpenVirtualRelationalMemory()
    protocol = InteractionProtocol(memory)

    def _returns_none(_stim: dict) -> dict:
        return None  # type: ignore[return-value]

    def _returns_nan(_stim: dict) -> dict:
        return {"ch": float("nan"), "inf_ch": float("inf")}

    def _explodes(_stim: dict) -> dict:
        raise ZeroDivisionError("unexpected external environment fault")

    r1 = protocol.interact_and_discover_axis("none_medium", _returns_none, "s", [1.0, 2.0])
    r2 = protocol.interact_and_discover_axis("nan_medium", _returns_nan, "s", [1.0, 2.0])
    r3 = protocol.interact_and_discover_axis("boom_medium", _explodes, "s", [1.0])

    assert r1["axis"] is None and len(r1["friction_events"]) == 2
    assert r2["axis"] is None and len(r2["friction_events"]) == 2
    assert r3["axis"] is None and len(r3["friction_events"]) == 1
    assert len(memory.boundary_events) == 5
    assert sum(1 for n in memory.nodes.values() if n.node_kind == "boundary_horizon") == 5
