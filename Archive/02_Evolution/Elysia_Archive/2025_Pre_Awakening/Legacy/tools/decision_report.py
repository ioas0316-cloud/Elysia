"""
Decision Report Utility

Creates a lightweight decision report node in the KG capturing:
- kind (journal/creative/math_verify/...)
- reason (natural language)
- confidence (float)
- gains/tradeoffs (lists of short texts)
- evidence_paths (list of file paths)

This keeps decisions explainable without changing core KGManager.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional, Dict, Any
from tools.kg_manager import KGManager


def _ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


def create_decision_report(
    kg: KGManager,
    kind: str,
    reason: str,
    confidence: float,
    result: Optional[Dict[str, Any]] = None,
    gains: Optional[Iterable[str]] = None,
    tradeoffs: Optional[Iterable[str]] = None,
    evidence_paths: Optional[Iterable[str]] = None,
) -> str:
    node_id = f"decision_report_{_ts()}"
    props = {
        "type": "decision_report",
        "kind": kind,
        "reason": reason,
        "confidence": float(confidence),
        "gains": list(gains or []),
        "tradeoffs": list(tradeoffs or []),
        "evidence_paths": list(evidence_paths or []),
        "result": result or {},
        "timestamp": _ts(),
    }
    kg.add_node(node_id, properties=props)
    return node_id

