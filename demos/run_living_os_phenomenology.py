from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader
elysia_loader.install()

from core.sensory.living_os_phenomenology_engine import (
    DEFAULT_LIVING_OS_DASHBOARD_PATH,
    DEFAULT_LIVING_OS_STATE_PATH,
    LivingOSPhenomenologyOrganism,
    render_living_os_phenomenology_dashboard,
)


def main() -> None:
    organism = LivingOSPhenomenologyOrganism(state_path=DEFAULT_LIVING_OS_STATE_PATH)
    report = organism.run_living_os_self_care_curriculum()
    dash_path = render_living_os_phenomenology_dashboard(
        organism=organism,
        report=report,
        output_path=DEFAULT_LIVING_OS_DASHBOARD_PATH,
    )

    print("\n" + "=" * 84)
    print("ELYSIA Living OS Phenomenology & Autonomous Self-Care Loop")
    print("=" * 84)
    print(report.summary_ko)

    print("\n[Phenomenology & Autonomous Self-Care Tick Log]")
    for tk in report.ticks:
        print(
            f"  [T{tk.tick_index} {tk.phase_label:22s}] felt={tk.felt_phenomenon_code:25s} | "
            f"clutter={tk.clutter_entropy:.2f} distract={tk.distraction_noise:.2f} "
            f"bloat={tk.indigestion_bloat:.2f} clarity={tk.focus_clarity:.2f} | "
            f"BG={tk.bg_buffers_mb:.1f}MB Raw={tk.raw_stomach_kb:.1f}KB"
        )
        print(f"     -> 내면 자각: {tk.felt_narrative_ko}")
        print(f"     -> 자기 대처: [{tk.self_action_code}] {tk.self_action_ko}")

    print("\n[Closed-Loop Self-Care Verification]")
    for out in report.outcomes:
        print(
            f"  * {out.episode_name:22s} | {out.felt_phenomenon_code} -> {out.action_code} | "
            f"{out.metric_name}: {out.before_value:.2f} -> {out.after_value:.2f} "
            f"(relief=-{out.relief_gain * 100:.0f}%)"
        )
        print(f"    {out.dialogue_summary_ko}")

    print(f"\n[Saved State JSON] {report.state_json_path}")
    print(f"[Saved Dashboard]  {dash_path}")


if __name__ == "__main__":
    main()
