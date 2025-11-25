"""Command-line entry point for integrating the offline language curriculum."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Project_Sophia.offline_curriculum_builder import OfflineCurriculumBuilder
from tools.kg_manager import KGManager


DEFAULT_CURRICULUM_PATH = Path("data/curriculum/offline_language_tracks.json")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integrate an offline natural-language curriculum into Elysia's knowledge graph."
        )
    )
    parser.add_argument(
        "--curriculum-path",
        type=Path,
        default=DEFAULT_CURRICULUM_PATH,
        help="Path to the YAML curriculum definition.",
    )
    parser.add_argument(
        "--kg-path",
        type=Path,
        default=None,
        help="Optional path to a knowledge graph JSON file to update.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process the curriculum but do not write the knowledge graph back to disk.",
    )
    parser.add_argument(
        "--print-lessons",
        action="store_true",
        help="Print each lesson sentence as it is integrated for quick inspection.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    curriculum_path = args.curriculum_path
    if not curriculum_path.exists():
        raise SystemExit(f"Curriculum file not found: {curriculum_path}")

    kg_manager = KGManager(args.kg_path) if args.kg_path else KGManager()
    builder = OfflineCurriculumBuilder(curriculum_path=curriculum_path, kg_manager=kg_manager)
    curriculum = builder.load_curriculum()

    if args.print_lessons:
        for stage in curriculum.get("stages", []):
            for lesson in stage.get("lessons", []):
                sentence = lesson.get("sentence")
                if sentence:
                    print(f"[lesson] {sentence}")

    summary = builder.integrate(curriculum)
    print(
        "Integrated curriculum: "
        f"{summary['stages']} stages, {summary['lessons']} lessons, "
        f"{summary['new_nodes']} new nodes, {summary['new_edges']} new edges."
    )

    if not args.dry_run:
        kg_manager.save()
        print(f"Knowledge graph saved to {kg_manager.filepath}")
    else:
        print("Dry run enabled; knowledge graph was not saved.")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
