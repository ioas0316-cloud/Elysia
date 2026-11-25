r"""
ELYSIA Uncontrolled Autonomous Life Runner
==========================================

Zero scripted prompts, zero developer-supplied answers, zero user commands.
Runs `SelfModifyingCausalOrganism` directly against the live PC desktop & workspace:
  Observe S_t -> Autonomous Action -> Predict S_{t+1} -> Compare Reality (What/How much/How/Why)
  -> Self-Modify (weights / new perceptual axis / policy / rollback) -> Persist & Repeat.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import elysia_loader

elysia_loader.install()

from core.embodiment.pc_sensory_motor_body import (  # noqa: E402
    ScreenRetinaSensor,
    WindowsForegroundSensor,
)
from core.embodiment.self_modifying_organism import SelfModifyingCausalOrganism  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Elysia Uncontrolled Self-Modifying Life Runner")
    parser.add_argument("--cycles", type=int, default=12, help="Number of autonomous live cycles")
    parser.add_argument("--interval-s", type=float, default=0.1, help="Pause between cycles")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=_ROOT / "data" / "elysia_autonomous_life",
        help="Persistent organism state directory",
    )
    parser.add_argument("--fresh", action="store_true", help="Reset prior organism weights")
    args = parser.parse_args()

    state_file = args.state_dir / "self_modifying_organism.json"
    if args.fresh and args.state_dir.exists():
        shutil.rmtree(args.state_dir, ignore_errors=True)

    org = SelfModifyingCausalOrganism.load(state_file)
    retina = ScreenRetinaSensor()
    fg = WindowsForegroundSensor()

    print("=" * 92)
    print(f"  ELYSIA UNCONTROLLED AUTONOMOUS LIFE (cycles={args.cycles}, initial_axes={len(org.axis_names)})")
    print("=" * 92)

    errors: list[float] = []
    for _ in range(max(1, args.cycles)):
        tr = org.run_autonomous_cycle(retina=retina, fg_sensor=fg, workspace_root=_ROOT)
        errors.append(tr.contrast.how_much_l2)
        c = tr.contrast
        m = tr.modification
        top_deltas = sorted(c.how_signed_deltas.items(), key=lambda kv: -abs(kv[1]))[:3]
        delta_str = ", ".join(f"{k}:{v:+.4f}" for k, v in top_deltas)
        print(
            f"  [Cycle {tr.cycle:02d}] act={tr.action_chosen:22s} | win='{tr.window_title[:18]}' | "
            f"err={c.how_much_l2:.4f} -> reobs={tr.reobserved_error_norm:.4f}"
        )
        print(
            f"     What(이탈 {len(c.what_diverged)}축) | How({delta_str}) | Why={c.why_attribution}"
        )
        print(
            f"     SelfMod={m.mutation_kind:16s} | dW={m.parameter_delta_norm:.4f} | "
            f"과거재평가: {m.past_error_before:.4f} -> {m.past_error_after:.4f} | "
            f"축수={tr.active_axes_count} | 유지={m.kept} (반사실최선={m.counterfactual_best_action})"
        )
        org.save(state_file)
        if args.interval_s > 0:
            time.sleep(args.interval_s)

    first_mean = float(np.mean(errors[: min(3, len(errors))]))
    last_mean = float(np.mean(errors[-min(3, len(errors)) :]))
    print("\n" + "-" * 92)
    print(f"  [자율 변화 요약] 초기 3턴 평균오차={first_mean:.4f} -> 최근 3턴 평균오차={last_mean:.4f}")
    print(f"  [구조 개변] 활성 지각축={len(org.axis_names)}개 ({org.axis_names}) | 유지={org.mutations_kept} | 롤백={org.mutations_reverted}")
    print(f"  [자율 정책 가치] {org.action_policy_values}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
