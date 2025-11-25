import argparse
import json
import os
import sys


def _ensure_repo_root_on_path() -> str:
    """Ensure the repository root is on sys.path and return it."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Elysia's feeling state from world events."
    )
    parser.add_argument(
        "--events",
        type=str,
        default="logs/world_events.jsonl",
        help="Path to raw world event log.",
    )
    parser.add_argument(
        "--signals",
        type=str,
        default="logs/elysia_signals.jsonl",
        help="Path to Elysia signal log (will be overwritten).",
    )
    parser.add_argument(
        "--half_life",
        type=float,
        default=200.0,
        help="Half-life (in ticks) for feeling decay.",
    )
    args = parser.parse_args()

    _ensure_repo_root_on_path()

    # Import after sys.path is prepared.
    from Project_Sophia.core.elysia_signal_engine import ElysiaSignalEngine
    from ELYSIA.CORE.feeling_buffer import ElysiaFeelingBuffer

    # 1) Raw events -> ElysiaSignalLog
    engine = ElysiaSignalEngine(
        raw_log_path=args.events,
        signal_log_path=args.signals,
    )
    engine.generate_signals_from_log()

    # 2) ElysiaSignalLog -> FeelingBuffer
    buffer = ElysiaFeelingBuffer(half_life_ticks=args.half_life)
    buffer.load_from_log(args.signals)

    raw_state = buffer.current_state().as_dict()
    squashed = buffer.squashed_state()

    print(
        json.dumps(
            {
                "raw_feelings": raw_state,
                "squashed_feelings": squashed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

