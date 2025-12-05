"""CLI for running the virtual caretaker reinforcement learning simulator."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from Core.Foundation.caretaker_rl_simulator import run_training


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a nurturing caretaker policy in a virtual environment.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=80,
        help="Number of training episodes to run (default: 80).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum caretaker actions per episode (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reports/nurturing"),
        help="Directory where the Markdown report will be saved.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist the report to disk; only print a summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = run_training(episodes=args.episodes, max_steps=args.max_steps, seed=args.seed)

    head, tail = stats.reward_trend()
    print("=== Nurturing Caretaker RL Training Summary ===")
    print(f"Episodes: {args.episodes}")
    print(f"Avg reward (first 10): {head:.3f}")
    print(f"Avg reward (last 10): {tail:.3f}")
    print("--- Greedy policy demo ---")
    for step in stats.greedy_episode.steps:
        print(f"[{step.step_index}] {step.action} -> {step.child_output}")

    if args.no_save:
        return

    report_dir = args.output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"caretaker_rl_report_{timestamp}.md"
    report_path.write_text(stats.to_report(), encoding="utf-8")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()

