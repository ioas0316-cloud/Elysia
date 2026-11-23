from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader
elysia_loader.install()

from core.sensory.elysia_unified_mind_organism import (
    DEFAULT_UNIFIED_MIND_DASHBOARD_PATH,
    DEFAULT_UNIFIED_STATE_PATH,
    ElysiaUnifiedMindOrganism,
    render_unified_mind_dashboard,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ELYSIA Unified Mind Organism: End-to-End Live Cycle & Meta-Causal Self-Introspection"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "screen"],
        default="full",
        help="full: Pygame embodied arena + real camera MP4 | screen: Pygame + live PC desktop observer via mss",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Capture duration in seconds when --mode screen is selected",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Capture FPS when --mode screen is selected",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Show live Pygame arena window instead of headless capture",
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default=str(DEFAULT_UNIFIED_STATE_PATH),
        help="Output JSON path for persistent causal mind state",
    )
    parser.add_argument(
        "--dashboard-path",
        type=str,
        default=str(DEFAULT_UNIFIED_MIND_DASHBOARD_PATH),
        help="Output PNG path for the 4-panel unified mind dashboard",
    )
    args = parser.parse_args()

    organism = ElysiaUnifiedMindOrganism(state_path=Path(args.state_path))
    report = organism.live_unified_cycle(
        mode=args.mode,
        screen_seconds=args.seconds,
        screen_fps=args.fps,
        headless=not args.show_window,
    )
    dash_path = render_unified_mind_dashboard(
        organism=organism,
        report=report,
        output_path=Path(args.dashboard_path),
    )

    print("\n" + "=" * 72)
    print(report.cycle_summary_ko)
    print("=" * 72)

    print("\n[Phase 3: Visual-Linguistic Layer Superposition Verdicts]")
    for r in report.contrast_results:
        print(
            f"  - {r.entity_id:12s} vs {r.statement_id:16s} -> {r.verdict:24s} | "
            f"sync(u/m/M)=({r.sync_micro:.2f}/{r.sync_meso:.2f}/{r.sync_macro:.2f}) | "
            f"grounded={r.grounded_super_molecule_id}"
        )

    print("\n[Phase 4: Analogical Foothold Reflection]")
    for ar in report.analogical_reports:
        print(
            f"  - {ar.target_domain_id:26s} | breaks={ar.breakdown_probes} | "
            f"torn={ar.torn_analogical_beams} | "
            f"accuracy={ar.naive_analogy_accuracy * 100:.0f}% -> {ar.reinterpreted_accuracy * 100:.0f}%"
        )

    print("\n[Meta-Causal Self-Introspection: Why Did ELYSIA's Eye & Mind Judge So?]")
    for key in (
        "ent_1::stmt_red_wall",
        "ent_3::stmt_red_wall",
        "ent_unseen_a::stmt_red_wall",
        "Invisible_Electricity",
        "Invisible_Heat_Conduction",
    ):
        exp = organism.introspect_why_i_judged(key)
        print(f"\n  >> {key}")
        print(f"     Chain ({len(exp.causal_chain_ids)} steps): {' -> '.join(exp.causal_chain_ids)}")
        print(f"     {exp.narrative_ko}")

    print(f"\n[Saved State JSON] {report.state_json_path}")
    print(f"[Saved Dashboard]  {dash_path}")


if __name__ == "__main__":
    main()
