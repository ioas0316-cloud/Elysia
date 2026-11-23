from __future__ import annotations

import json
from pathlib import Path

from core.sensory.cellular_world_communication_protocol import (
    DEFAULT_CELLULAR_DASHBOARD_PATH,
    DEFAULT_CELLULAR_STATE_PATH,
    CellularWorldDialogueField,
    LivingWorldCell,
    render_cellular_world_dialogue_dashboard,
)


def test_scale_invariant_cellular_world_communication_protocol() -> None:
    """
    Verifies ELYSIA's single scale-invariant `LivingWorldCell` & `converse_until_sync`
    engine operating WITHOUT per-domain pipelines across 3 scales inside ONE `CausalField`:
      1. Scale 0 (Cell <-> Cell): Eye_Cell & Touch_Cell synchronize via membrane friction
         into a unified optical illusion signature.
      2. Scale 1 (Organism World <-> Language World): Elysia_Organism_World consolidated
         from its sub-cells REFUSES false overwrite by 'Red Wall' (`synced=False`) and
         reaches consensus with truthful 'Red Illusion' (`synced=True`).
      3. Scale 2 (Teacher World <-> Student World): Water-anchored ELYSIA converses with
         Teacher_Electricity_World and unseen Teacher_Magnetism_World, lifting unseen
         counterfactual accuracy 50% -> 100% on both while preserving shared invariants.
      4. Persists dialogue telemetry JSON and renders 4-panel visual dashboard.
    """
    field_sys = CellularWorldDialogueField(state_path=DEFAULT_CELLULAR_STATE_PATH)
    res = field_sys.run_three_scale_living_curriculum()

    # All interacting entities are genuinely instances of ONE single universal class
    assert all(isinstance(c, LivingWorldCell) for c in field_sys.cells.values())

    print("\n=== [ELYSIA Cellular World Communication Verification Log] ===")
    for k in (
        "scale0_dialogue",
        "scale1_reject_dialogue",
        "scale1_sync_dialogue",
        "scale2_elec_dialogue",
        "scale2_mag_dialogue",
    ):
        print(f"  {res[k].summary_ko}")

    # Scale 0: Eye_Cell <-> Touch_Cell reach genuine reciprocal sync
    d0 = res["scale0_dialogue"]
    assert d0.synced is True
    assert d0.final_alignment >= 0.90
    assert d0.beam_strength >= 0.85

    # Scale 1: Organism world grounded in Eye+Touch refuses false 'RedWall' symbol
    # and reaches full sync on truthful 'RedIllusion' symbol
    d1_rej = res["scale1_reject_dialogue"]
    d1_sync = res["scale1_sync_dialogue"]
    assert d1_rej.synced is False
    assert d1_rej.final_alignment < 0.72
    assert d1_rej.beam_strength < 0.25
    assert d1_sync.synced is True
    assert d1_sync.final_alignment >= 0.92
    assert d1_sync.beam_strength >= 0.85

    # Scale 2: Teacher Electricity & Unseen Magnetism lift counterfactual accuracy 50% -> 100%
    d2_elec = res["scale2_elec_dialogue"]
    d2_mag = res["scale2_mag_dialogue"]
    assert d2_elec.synced is True and d2_elec.final_alignment >= 0.90
    assert d2_mag.synced is True and d2_mag.final_alignment >= 0.90
    acc = res["accuracy"]
    assert acc["electricity_before"] == 0.5
    assert acc["electricity_after"] == 1.0
    assert acc["magnetism_before"] == 0.5
    assert acc["magnetism_after"] == 1.0

    # Shared 6D CausalField holds all cells & inter-scale beams
    assert res["causal_field_voxels"] == len(field_sys.cells)
    assert res["causal_field_beams"] >= 7

    # Introspection verifies both preserved lived foothold dimensions and dialogue-revised dims
    ins_elec = res["introspections"]["Elysia_Mind_Elec"]
    ins_mag = res["introspections"]["Elysia_Mind_Mag"]
    assert "d2_contact_resist" in ins_elec.preserved_dims
    assert "d4_gradient_drive" in ins_elec.preserved_dims
    assert any("d3_cut_persist" in s for s in ins_elec.revised_dims)
    assert any("d5_propagation_mode" in s for s in ins_elec.revised_dims)
    assert any("d3_cut_persist" in s for s in ins_mag.revised_dims)
    assert any("d5_propagation_mode" in s for s in ins_mag.revised_dims)

    # Persistent JSON validation
    state_file = Path(res["state_json_path"])
    assert state_file.exists() and state_file.stat().st_size > 2_000
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["accuracy"]["electricity_after"] == 1.0
    assert saved["accuracy"]["magnetism_after"] == 1.0
    assert len(saved["dialogues"]) == 5

    # Dashboard rendering validation
    dash_path = render_cellular_world_dialogue_dashboard(
        field_sys=field_sys,
        curriculum_result=res,
        output_path=DEFAULT_CELLULAR_DASHBOARD_PATH,
    )
    assert dash_path.exists() and dash_path.stat().st_size > 20_000
