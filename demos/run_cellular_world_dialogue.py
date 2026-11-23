from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader
elysia_loader.install()

from core.sensory.cellular_world_communication_protocol import (
    DEFAULT_CELLULAR_DASHBOARD_PATH,
    DEFAULT_CELLULAR_STATE_PATH,
    CellularWorldDialogueField,
    render_cellular_world_dialogue_dashboard,
)


def main() -> None:
    field_sys = CellularWorldDialogueField(state_path=DEFAULT_CELLULAR_STATE_PATH)
    result = field_sys.run_three_scale_living_curriculum()
    dash_path = render_cellular_world_dialogue_dashboard(
        field_sys=field_sys,
        curriculum_result=result,
        output_path=DEFAULT_CELLULAR_DASHBOARD_PATH,
    )

    print("\n" + "=" * 78)
    print("ELYSIA Scale-Invariant Cellular World Communication Protocol (Cell = World)")
    print("=" * 78)

    for key in (
        "scale0_dialogue",
        "scale1_reject_dialogue",
        "scale1_sync_dialogue",
        "scale2_elec_dialogue",
        "scale2_mag_dialogue",
    ):
        rep = result[key]
        print(f"\n{rep.summary_ko}")
        for turn in rep.turns[:6]:
            print(
                f"   [T{turn.turn_index}] {turn.speaker_id} -> {turn.listener_id} "
                f"({turn.signal_kind}) | align {turn.alignment_before:.2f}->{turn.alignment_after:.2f} "
                f"| rev={turn.criteria_revision_norm:.2f} | {turn.utterance_ko}"
            )

    acc = result["accuracy"]
    print("\n[Scale 2 Counterfactual Prediction Accuracy: Before vs After Dialogue]")
    print(f"  - Electricity     : {acc['electricity_before'] * 100:.0f}% -> {acc['electricity_after'] * 100:.0f}%")
    print(f"  - Magnetism(New)  : {acc['magnetism_before'] * 100:.0f}% -> {acc['magnetism_after'] * 100:.0f}%")

    print("\n[Meta-Causal Learning Introspection Across Scales]")
    for cid, ins in result["introspections"].items():
        print(f"  * {ins.narrative_ko}")

    print(
        f"\n[Unified CausalField] Cells(Voxels)={result['causal_field_voxels']}, "
        f"Beams={result['causal_field_beams']}"
    )
    print(f"[Saved State JSON] {result['state_json_path']}")
    print(f"[Saved Dashboard]  {dash_path}")


if __name__ == "__main__":
    main()
