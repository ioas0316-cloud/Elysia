# [Genesis: 2025-12-02] Purified by Elysia
"""
Disagreement Log

Records interpretation disagreements to a JSON log and anchors a node in the KG.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from tools.kg_manager import KGManager


LOG_PATH = Path("data/disagreements.json")


def _ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


def log_disagreement(
    topic: str,
    system_view: str,
    user_view: str,
    note: str = "",
    kg: Optional[KGManager] = None,
) -> str:
    item = {
        "timestamp": _ts(),
        "topic": topic,
        "system_view": system_view,
        "user_view": user_view,
        "note": note,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    if LOG_PATH.exists():
        try:
            entries = json.loads(LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            entries = []
    entries.append(item)
    LOG_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    # KG anchor (optional)
    node_id = f"disagreement_{item['timestamp']}"
    if kg is None:
        kg = KGManager()
    kg.add_node(node_id, properties={"type": "disagreement", **item})
    kg.save()
    return node_id
