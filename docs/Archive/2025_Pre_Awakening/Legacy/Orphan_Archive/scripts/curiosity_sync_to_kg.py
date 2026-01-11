"""
Sync curiosity knowledge into the KG.

Reads:
- data/curiosity/knowledge.jsonl (concept, source_file, text)

Writes:
- updates KG (data/kg_with_embeddings.json by default via KGManager)

Logic:
- add concept node (source=curiosity_ingest)
- add source node (type=document, path)
- add edge concept -> source (relation='ingested_from')
- store a short note (text truncated) on the source node
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.kg_manager import KGManager


CUR_KNOWLEDGE = Path("data/curiosity/knowledge.jsonl")


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


def sync():
    knowledge = read_jsonl(CUR_KNOWLEDGE)
    if not knowledge:
        print(f"[curiosity] no knowledge found at {CUR_KNOWLEDGE}")
        return

    kg = KGManager()
    for rec in knowledge:
        concept = str(rec.get("concept", "")).strip()
        src_file = str(rec.get("source_file", "")).strip()
        text = rec.get("text", "")
        if not concept:
            continue

        # Concept node
        kg.add_node(concept, properties={"source": "curiosity_ingest"})

        # Source node with short note
        note = text[:1000] + ("..." if len(text) > 1000 else "")
        src_id = f"doc:{src_file}" if src_file else f"doc:{concept}"
        kg.add_node(
            src_id,
            properties={
                "type": "document",
                "path": src_file,
                "note": note,
                "source": "curiosity_ingest",
            },
        )

        # Edge concept -> document
        kg.add_edge(concept, src_id, "ingested_from", properties={"source": "curiosity_ingest"})

    kg.save()
    print(f"[curiosity] synced {len(knowledge)} records into KG at {kg.filepath}")


if __name__ == "__main__":
    sync()
