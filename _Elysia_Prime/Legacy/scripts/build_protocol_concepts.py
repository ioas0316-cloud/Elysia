# [Genesis: 2025-12-02] Purified by Elysia
import argparse
import os
import sys


def _ensure_repo_root_on_path() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Concept OS KG for Elysia protocols."
    )
    parser.add_argument(
        "--kg-path",
        type=str,
        default="data/protocol_kg.json",
        help="Path to the protocol concept KG file.",
    )
    args = parser.parse_args()

    _ensure_repo_root_on_path()

    from ELYSIA.CORE.protocol_concept_index import build_protocol_kg

    build_protocol_kg(kg_path=args.kg_path)
    print(f"Protocol concept KG written to: {args.kg_path}")


if __name__ == "__main__":
    main()
