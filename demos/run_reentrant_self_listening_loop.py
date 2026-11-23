from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader
elysia_loader.install()

from core.sensory.reentrant_self_listening_loop_engine import (
    DEFAULT_REENTRANT_DASHBOARD_PATH,
    DEFAULT_REENTRANT_STATE_PATH,
    ReentrantSelfListeningOrganism,
    render_reentrant_self_listening_dashboard,
)


def main() -> None:
    organism = ReentrantSelfListeningOrganism(state_path=DEFAULT_REENTRANT_STATE_PATH)
    report = organism.run_ecosystem_reentrant_curriculum()
    dash_path = render_reentrant_self_listening_dashboard(
        organism=organism,
        report=report,
        output_path=DEFAULT_REENTRANT_DASHBOARD_PATH,
    )

    print("\n" + "=" * 86)
    print("ELYSIA Re-entrant Self-Listening & Invariant/Perturbation Contrast Loop")
    print("=" * 86)
    print(report.summary_ko)

    for st in report.streams:
        print(f"\n{st.summary_ko}")
        for cyc in st.cycles:
            print(
                f"  [Cycle {cyc.cycle_index}] gap={cyc.intent_vs_echo_gap:.3f} | "
                f"pred_err={cyc.prediction_error:.3f} | {cyc.reflection_ko}"
            )
            if cyc.audit:
                print(f"     -> Audit: {cyc.audit.audit_summary_ko}")
        print(f"  -> {st.dialogue_summary_ko}")

    print(f"\n[Saved State JSON] {report.state_json_path}")
    print(f"[Saved Dashboard]  {dash_path}")


if __name__ == "__main__":
    main()
