import argparse
import os
import shutil
import time
from pathlib import Path

CATEGORY_MAP = {
    ".txt": "literature",
    ".md": "dialogues",
    ".py": "code",
    ".js": "code",
    ".json": "stories",
    ".csv": "stories",
}


def classify(path: Path) -> str:
    ext = path.suffix.lower()
    return CATEGORY_MAP.get(ext, "stories")


def ingest(source: Path, root: Path, dry_run: bool) -> int:
    moved = 0
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        category = classify(path)
        target_dir = root / category
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / path.name
        if target_path.exists():
            target_path = target_dir / f"{int(time.time())}_{path.name}"
        print(f" Ingesting {path.name} -> {target_dir.name}")
        if not dry_run:
            shutil.copy2(path, target_path)
        moved += 1
    return moved


def main():
    parser = argparse.ArgumentParser(description="Ingest new corpus files and refresh language supply.")
    parser.add_argument("--source", type=str, default="data/corpus_incoming", help="Directory with new assets.")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without copying.")
    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"Source directory '{source_dir}' does not exist.")
        return

    root = Path("data/corpus")

    count = ingest(source_dir, root, dry_run=args.dry_run)
    if count:
        print(f"✔ Ingested {count} file(s) into {root}.")
        if not args.dry_run:
            print("↻ Running meta agent to absorb new knowledge...")
            os.system("python scripts/run_meta_agent.py")
    else:
        print("— No files matched for ingestion.")


if __name__ == "__main__":
    main()
