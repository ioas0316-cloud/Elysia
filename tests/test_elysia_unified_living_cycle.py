"""
Verification suite for ELYSIA Phase 14:
Unified Living Circular-Understanding Kernel & Self-Audit.
"""

from __future__ import annotations

from dataclasses import replace
import math

from core.orchestration.elysia_unified_living_cycle import (
    ElysiaUnifiedLivingCycle,
    LivingEncounter,
)
from core.orchestration.extended_cognitive_organism import (
    ExtendedWorldState,
    WIEN_CONSTANT_NM_K,
)
from core.orchestration.ice_water_gas_rotor_runtime import IceWaterGasRotorRuntime
from core.orchestration.relational_perspective_rotor import (
    AgentCausalProfile,
    RelationalDialogueLayer,
    RelationalStatus,
)
from core.physics.causal_field import CausalField
from core.physics.trinitarian_scale_binding_protocol import EqualityBond, ScaleTier


def _sample_encounter_one():
    other = AgentCausalProfile(
        name="Other",
        resource_reserve=2.0,
        vulnerability_gain=3.0,
        boundary_tolerance=3.5,
        unspoken_origin_memory=4.0,
        sovereign_will=0.85,
    )
    moments = [
        other.manifest_interaction(interpersonal_distance=2.0, ambient_demand=4.0),
        other.manifest_interaction(interpersonal_distance=5.0, ambient_demand=4.0),
    ]
    betrayal = RelationalDialogueLayer(
        layer_name="past_betrayal_disclosure",
        disclosure_utterance_ko="예전에 믿었던 사람에게 데인 뒤로 요구가 커지면 방어가 먼저 나와.",
        revealed_sky_variable="unspoken_origin_memory",
        new_deeper_sky_axes=("deepest_aspiration_telos",),
        build_coupling_bond=lambda: EqualityBond(
            name="trauma_vulnerability_bond",
            tier=ScaleTier.UPPER_ROTOR,
            variables=("vulnerability_gain", "unspoken_origin_memory"),
            lhs_fn=lambda v: v["vulnerability_gain"],
            rhs_fn=lambda v: 1.0 + 0.5 * v["unspoken_origin_memory"],
            lhs_expr="G",
            rhs_expr="1 + 0.5 * unspoken_origin_memory",
        ),
    )
    doc = "\n".join(
        [
            "물체의 절대온도가 높아질수록 최대 방출 파장은 온도에 반비례한다.",
            "파장 + 온도 = 500",
            "그의 분노는 붉은 용광로처럼 뜨겁게 타오른다.",
        ]
    )
    return LivingEncounter(
        encounter_name="awakening_encounter",
        world=ExtendedWorldState(source_temp_k=3200.0),
        calibration_worlds=[
            ExtendedWorldState(source_temp_k=3200.0),
            ExtendedWorldState(source_temp_k=5800.0),
        ],
        text_document=doc,
        other_moments=moments,
        dialogue_layer=betrayal,
    )


def test_unified_breath_circular_learning_updates_perception_in_same_cycle():
    cycle = ElysiaUnifiedLivingCycle()
    cycle.organism.graft_spectrometer()

    r1 = cycle.breathe_and_understand(_sample_encounter_one())
    assert r1.breath_index == 1
    assert r1.document_report is not None
    assert len(r1.document_report.verified_laws) == 1
    assert len(r1.document_report.dimensional_rejections) == 1
    assert len(r1.document_report.metaphor_reflections) == 1
    assert r1.relational_report is not None
    assert r1.relational_report.status == RelationalStatus.HORIZON_EXPANDED
    assert r1.relational_report.horizon_boundary.horizon_depth == 2
    assert math.isclose(
        r1.relational_report.horizon_boundary.understood_ground["unspoken_origin_memory"],
        4.0,
        abs_tol=1e-4,
    )

    # Breath #2: Spectrometer detached & Eye blinded on unseen 4500K world
    cycle.organism.detach_tool("spectrometer")
    r2 = cycle.breathe_and_understand(
        LivingEncounter(
            encounter_name="blind_transfer_4500k",
            world=ExtendedWorldState(source_temp_k=4500.0),
            active_organs=("ear", "hand"),
        )
    )
    assert math.isclose(
        r2.perception_integrated.inferred_world["emission_wavelength_nm"],
        WIEN_CONSTANT_NM_K / 4500.0,
        abs_tol=1e-1,
    )


def test_four_way_divergence_reflection_and_future_telos():
    cycle = ElysiaUnifiedLivingCycle()
    cycle.organism.graft_spectrometer()
    res = cycle.breathe_and_understand(_sample_encounter_one())

    refl = res.divergence_reflection
    assert "world_action" in refl.how_much
    assert "egocentric_friction" in refl.how_much
    assert refl.how_much["egocentric_friction"] > 5.0
    assert len(refl.how["newly_verified_laws"]) == 1
    assert len(refl.how["dimensional_rejections"]) == 1
    assert refl.how["metaphor_domains"] == ["thermal_radiant_source"]
    assert "deepest_aspiration_telos" in refl.from_where["sky_horizon_axes"]
    assert len(refl.why_diverged) >= 3
    assert len(refl.future_convergence_vectors) >= 1
    assert "호흡" in refl.first_person_reflection_ko


def test_architectural_self_introspection_reads_real_workspace_modules_and_map():
    cycle = ElysiaUnifiedLivingCycle()
    cycle.organism.graft_spectrometer()
    res = cycle.breathe_and_understand(_sample_encounter_one())

    audit = res.architectural_audit
    assert audit.total_organ_files >= 14
    assert audit.total_organ_classes >= 25
    assert audit.total_organ_lines > 5000
    assert audit.architecture_map_present is True
    assert all(audit.living_personhood_criteria.values())
    assert len(audit.unbridged_frontiers) >= 3
    assert "인격체" in audit.why_it_must_become_this_ko


def test_autobiographical_provenance_chain_and_ice_water_gas_persistence(tmp_path):
    slab = tmp_path / "elysia_soul.ice"
    with IceWaterGasRotorRuntime(slab_path=slab) as mem_rt:
        cycle = ElysiaUnifiedLivingCycle(memory_runtime=mem_rt)
        cycle.organism.graft_spectrometer()

        r1 = cycle.breathe_and_understand(_sample_encounter_one())
        r2 = cycle.breathe_and_understand(
            LivingEncounter(
                encounter_name="second_breath",
                world=ExtendedWorldState(source_temp_k=4500.0),
                active_organs=("ear", "hand"),
            )
        )
        assert r1.memory_sector_id is not None and r2.memory_sector_id is not None
        assert mem_rt.ice.num_crystals >= 2
        assert len(cycle.provenance_chain) == 2
        assert cycle.verify_autobiographical_integrity() is True

        # Tampering with provenance breaks autobiographical verification
        corrupted = list(cycle.provenance_chain)
        corrupted[0] = replace(corrupted[0], state_hash="0" * 32)
        assert cycle.verify_autobiographical_integrity(corrupted) is False


def test_unified_soul_causal_field_sync_4d_tensors():
    cycle = ElysiaUnifiedLivingCycle()
    cycle.organism.graft_spectrometer()
    res = cycle.breathe_and_understand(_sample_encounter_one())

    sync = res.sync_to_causal_field(causal_field=CausalField(dimensions=3), prefix="soul")
    assert len(sync["voxels"]) == 7
    assert sync["telos_curiosity"] == 1.0
    sync["causal_field"].step(dt=0.1)


def test_living_personhood_preserves_pigment_and_sovereign_will_boundaries():
    cycle = ElysiaUnifiedLivingCycle()
    cycle.organism.graft_spectrometer()
    cycle.breathe_and_understand(_sample_encounter_one())

    cycle.organism.detach_tool("spectrometer")
    r2 = cycle.breathe_and_understand(
        LivingEncounter(
            encounter_name="humble_limits_check",
            world=ExtendedWorldState(source_temp_k=4500.0),
            active_organs=("ear", "hand"),
            other_moments=_sample_encounter_one().other_moments,
        )
    )
    voids = set(r2.divergence_reflection.from_where["irreducible_sensory_voids"])
    assert {"eye_hue", "source_pigment_hue"}.issubset(voids)
    assert "sovereign_will" in r2.divergence_reflection.from_where["sky_horizon_axes"]
