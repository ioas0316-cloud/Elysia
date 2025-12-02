# [Genesis: 2025-12-02] Purified by Elysia
"""
Ingest story texts into the KG as experience nodes.

Usage:
  python -m scripts.ingest_stories --path data/corpus/stories/sample_story.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from tools.kg_manager import KGManager


def ingest_story(path: Path, kg: KGManager):
    sid = f"story_{path.stem}"
    kg.add_node(sid, properties={"type": "story", "path": str(path)})
    # Optionally create a few chapter/line nodes
    lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    for i, line in enumerate(lines[:50], start=1):  # cap to avoid explosion
        text = line.strip()
        if not text:
            continue
        nid = f"{sid}_line_{i}"
        kg.add_node(nid, properties={"type": "story_line", "text": text})
        kg.add_edge(sid, nid, "has_line")
    kg.save()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', required=True)
    args = p.parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    kg = KGManager()
    ingest_story(path, kg)
    print('Ingested story from', path)


if __name__ == '__main__':
    main()
