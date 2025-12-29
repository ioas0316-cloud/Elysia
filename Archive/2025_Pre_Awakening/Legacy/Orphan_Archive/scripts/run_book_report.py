"""
Create a simple offline book report from a local text file. Extracts a
summary, candidate characters, and themes; saves a markdown report and adds
key nodes/edges into the KG.

Usage:
  python -m scripts.run_book_report --book path/to/book.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from Core.FoundationLayer.Foundation.reading_coach import ReadingCoach
from tools.kg_manager import KGManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", required=True, help="Path to local text file")
    args = parser.parse_args()

    book_path = Path(args.book)
    if not book_path.exists():
        raise SystemExit(f"Book file not found: {book_path}")

    text = book_path.read_text(encoding="utf-8", errors="ignore")
    coach = ReadingCoach()
    summary = coach.summarize_text(text, max_sentences=7)
    characters = coach.extract_characters(text, top_k=8)
    themes = coach.extract_themes(text, top_k=5)
    cooccurs = coach.sentence_cooccurrence_pairs(text, characters)
    conflicts = coach.extract_conflicts(text, characters)

    out_dir = Path("data/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = book_path.stem
    report_path = out_dir / f"{stem}_report.md"

    md = [
        f"# Book Report: {stem}",
        "", "## Summary", summary or "(No summary)",
        "", "## Characters", *(f"- {c}" for c in (characters or ["(None)"])),
        "", "## Themes", *(f"- {t}" for t in (themes or ["(None)"])),
    ]
    report_path.write_text("\n".join(md), encoding="utf-8")

    kg = KGManager()
    book_node = f"book_{stem}"
    kg.add_node(book_node, properties={"type": "book", "report_path": str(report_path)})
    for c in characters:
        cid = f"character_{c}"
        kg.add_node(cid, properties={"type": "character"})
        kg.add_edge(book_node, cid, "features_character")
    for t in themes:
        tid = f"theme_{t}"
        kg.add_node(tid, properties={"type": "theme"})
        kg.add_edge(book_node, tid, "has_theme")
    # Co-occurrence graph
    for a, b in cooccurs:
        kg.add_edge(f"character_{a}", f"character_{b}", "co_occurs")
    # Conflict graph
    for i, cf in enumerate(conflicts, start=1):
        cn = f"conflict_{stem}_{i}"
        kg.add_node(cn, properties={"type": "conflict", "snippet": cf.get("snippet", "")})
        kg.add_edge(book_node, cn, "has_conflict")
        kg.add_edge(f"character_{cf['a']}", cn, "involved_in")
        kg.add_edge(f"character_{cf['b']}", cn, "involved_in")
    kg.save()

    print("Report saved:", report_path)


if __name__ == "__main__":
    main()
