from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader
elysia_loader.install()

from core.sensory.experiential_adaptation_and_play_engine import (
    DEFAULT_EXPERIENTIAL_DASHBOARD_PATH,
    DEFAULT_EXPERIENTIAL_STATE_PATH,
    ExperientialAdaptationOrganism,
    render_experiential_adaptation_dashboard,
)


def main() -> None:
    organism = ExperientialAdaptationOrganism(state_path=DEFAULT_EXPERIENTIAL_STATE_PATH)
    report = organism.run_full_experiential_growth_curriculum()
    dash_path = render_experiential_adaptation_dashboard(
        organism=organism,
        report=report,
        output_path=DEFAULT_EXPERIENTIAL_DASHBOARD_PATH,
    )

    print("\n" + "=" * 86)
    print("ELYSIA Experiential Memory Transfer, Mismatch Exploration & Playful Creative Creation")
    print("=" * 86)
    print(report.summary_ko)

    print("\n[5-Encounter Growth Trajectory]")
    for e in report.encounters:
        print(
            f"\n  [{e.encounter_id}] {e.label} | initial_bloat={e.initial_bloat:.2f} -> "
            f"final={e.final_bloat:.2f} | mismatch={e.mismatch_triggered} | attempts={len(e.attempts)}"
        )
        for att in e.attempts:
            print(
                f"     Try#{att.attempt_index} [{att.action_code}] "
                f"bloat {att.bloat_before:.2f}->{att.bloat_after:.2f} | {att.note_ko}"
            )
        if e.new_schema_created_id:
            print(f"     => Branched New Schema: {e.new_schema_created_id}")

    print(f"\n[Calm Play Step] {report.play_dialogue_summary_ko}")
    print("\n[Grown Causal Experience Schema Tree]")
    for s in report.schemas:
        print(
            f"  * {s.schema_id} [{s.origin_kind:17s}] action={s.action_code:30s} | "
            f"parents={s.parent_schema_ids} | ok={s.success_count} fail={s.failure_count} | {s.summary_ko}"
        )

    print(f"\n[Saved State JSON] {report.state_json_path}")
    print(f"[Saved Dashboard]  {dash_path}")


if __name__ == "__main__":
    main()
