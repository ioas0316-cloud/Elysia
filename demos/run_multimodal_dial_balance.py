from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader
elysia_loader.install()

from core.sensory.multimodal_dial_balance_inference_engine import (
    DEFAULT_DIAL_DASHBOARD_PATH,
    DEFAULT_DIAL_STATE_PATH,
    MultimodalDialBalanceEngine,
    render_multimodal_dial_dashboard,
)


def main() -> None:
    engine = MultimodalDialBalanceEngine(state_path=DEFAULT_DIAL_STATE_PATH)
    report = engine.run_multimodal_dial_curriculum()
    dash_path = render_multimodal_dial_dashboard(
        engine=engine,
        report=report,
        output_path=DEFAULT_DIAL_DASHBOARD_PATH,
    )

    print("\n" + "=" * 86)
    print("ELYSIA Multi-Dimensional Variable-Resistor Dial Equilibrium & Cross-Modal Inference")
    print("=" * 86)
    print(report.summary_ko)

    print("\n[Experiment 1: Blind Vision Reconstruction (m0_vision = Masked Completely)]")
    for res in report.blind_vision_results:
        print(f"  * {res.narrative_ko}")

    print("\n[Experiment 2: Exponential Error Collapse with Active Coupled Dials (k = 1 -> 5)]")
    for pt in report.scaling_curve:
        print(
            f"  - Active Dials={pt.active_dial_count} | Cross-Links={pt.combinatorial_pairs:2d} pairs | "
            f"Vision MAE={pt.mean_abs_error:.3f} | Uncertainty={pt.mean_uncertainty:.3f}"
        )

    print("\n[Experiment 3: Double-Blind Unseen Live Wire (BOTH Vision & Touch Masked!)]")
    print(f"  * {report.double_blind_result.narrative_ko}")

    print(f"\n[Saved State JSON] {report.state_json_path}")
    print(f"[Saved Dashboard]  {dash_path}")


if __name__ == "__main__":
    main()
