"""
Curiosity apply step: marks requests as fulfilled when ingested data exists,
and writes a normalized knowledge file for downstream use.

Inputs:
- data/curiosity/requests.jsonl (optional)
- data/curiosity/ingested.jsonl (required)

Outputs:
- data/curiosity/requests.resolved.jsonl (updated statuses)
- data/curiosity/knowledge.jsonl (concept, source_file, text)

No external dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path


CUR_DIR = Path("data/curiosity")
REQ_PATH = CUR_DIR / "requests.jsonl"
ING_PATH = CUR_DIR / "ingested.jsonl"
REQ_OUT = CUR_DIR / "requests.resolved.jsonl"
KNOWLEDGE_OUT = CUR_DIR / "knowledge.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in items:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def apply():
    ingested = read_jsonl(ING_PATH)
    requests = read_jsonl(REQ_PATH)

    # Build concept -> list of ingested records
    ing_by_concept: dict[str, list[dict]] = {}
    for rec in ingested:
        concept = str(rec.get("concept", "")).strip()
        if not concept:
            continue
        ing_by_concept.setdefault(concept, []).append(rec)

    # Update request statuses if ingested data exists
    resolved_requests = []
    for req in requests:
        concept = str(req.get("concept", "")).strip()
        if concept in ing_by_concept:
            req["status"] = "fulfilled"
        resolved_requests.append(req)

    # Normalize knowledge entries
    knowledge = []
    for concept, items in ing_by_concept.items():
        for rec in items:
            knowledge.append(
                {
                    "concept": concept,
                    "source_file": rec.get("file", ""),
                    "text": rec.get("text", ""),
                }
            )

    write_jsonl(REQ_OUT, resolved_requests)
    write_jsonl(KNOWLEDGE_OUT, knowledge)

    print(f"[curiosity] concepts ingested: {len(ing_by_concept)}")
    print(f"[curiosity] knowledge written: {KNOWLEDGE_OUT}")
    print(f"[curiosity] requests resolved: {REQ_OUT}")


if __name__ == "__main__":
    apply()
