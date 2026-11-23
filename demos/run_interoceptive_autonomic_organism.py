from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader
elysia_loader.install()

from core.sensory.interoceptive_autonomic_self_organism import (
    DEFAULT_INTEROCEPTIVE_DASHBOARD_PATH,
    DEFAULT_INTEROCEPTIVE_STATE_PATH,
    InteroceptiveAutonomicOrganism,
    render_interoceptive_autonomic_dashboard,
)


def main() -> None:
    organism = InteroceptiveAutonomicOrganism(state_path=DEFAULT_INTEROCEPTIVE_STATE_PATH)
    report = organism.run_four_episode_autonomic_life()
    dash_path = render_interoceptive_autonomic_dashboard(
        organism=organism,
        report=report,
        output_path=DEFAULT_INTEROCEPTIVE_DASHBOARD_PATH,
    )

    print("\n" + "=" * 82)
    print("ELYSIA Interoceptive Autonomic Nervous System & 4-Tier Scale Boundary Organism")
    print("=" * 82)
    print(report.summary_ko)

    print("\n[Physiological & Autonomic Trajectory Across 16 Live Host Ticks]")
    for s in report.snapshots:
        print(
            f"  [T{s.tick_index:02d} {s.episode_label:15s}] felt={s.felt_state_code:13s} | "
            f"CPU={s.cpu_percent:4.1f}% RAM={s.ram_percent:4.1f}% | "
            f"hunger={s.gut_hunger_signal:.2f} pain={s.pain_trauma_signal:.2f} | "
            f"symp={s.sympathetic_tone:.2f} para={s.parasympathetic_tone:.2f} open={s.membrane_openness:.2f}"
        )

    print("\n[Autonomous 4-Tier Scale Boundary Spectrum: Controllability vs Observability]")
    for b in report.boundary_map:
        print(
            f"  - {b.cell_id:26s} | ctrl={b.controllability:.2f}, obs={b.observability:.2f} "
            f"-> {b.inferred_tier:18s} ({b.tier_label_ko})"
        )

    print("\n[Felt-State Interoceptive Introspections]")
    for k, ins in report.introspections.items():
        print(f"  * {ins.narrative_ko}")

    print(f"\n[Saved State JSON] {report.state_json_path}")
    print(f"[Saved Dashboard]  {dash_path}")


if __name__ == "__main__":
    main()
