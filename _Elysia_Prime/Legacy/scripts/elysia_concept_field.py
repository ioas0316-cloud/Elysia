# [Genesis: 2025-12-02] Purified by Elysia
"""
Elysia concept field (v0)

Purpose
- Represent "what Elysia knows / is learning about" as a simple concept field,
  so that learning is not only pattern frequency but also concept-level change.

Layers
- Layer 1: episodes / patterns (SymbolEpisode, TextEpisode, expression scores).
- Layer 2: concept field (per symbol/word/world-concept entry with usage and feelings).
- Layer 3: meta layer (concept notes summarising "what changed" about a concept).

This module builds Layer 2 and leaves hooks for Layer 3.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List


KIND_CATHEDRAL_COORD = {
    "symbol": "S-L2-p",
    "word": "S-L2-p",
    "world_concept": "S-L2-e",
}


def _default_cathedral_coord(kind: str) -> str:
    return KIND_CATHEDRAL_COORD.get(kind, "S-L2-e")


@dataclass
class ConceptEntry:
    """Accumulated view of a single concept."""

    name: str
    kind: str  # "symbol", "word", "world_concept"
    usage_count: int = 0
    contexts: Dict[str, int] = field(default_factory=dict)  # e.g. {"symbol_lesson": 3}
    feelings: Dict[str, float] = field(default_factory=dict)  # averaged axes (JOY, CARE, etc.)
    examples: List[str] = field(default_factory=list)  # short example texts
    last_timestamp: int = 0
    cathedral_coord: str = "S-L2-e"

    def bump_usage(self, ctx: str, ts: int) -> None:
        self.usage_count += 1
        self.contexts[ctx] = self.contexts.get(ctx, 0) + 1
        if ts > self.last_timestamp:
            self.last_timestamp = ts

    def update_feelings(self, axes: Dict[str, float]) -> None:
        # Simple running average merge for overlapping keys.
        for k, v in axes.items():
            v = float(v)
            if k in self.feelings:
                self.feelings[k] = (self.feelings[k] * (self.usage_count - 1) + v) / max(self.usage_count, 1)
            else:
                self.feelings[k] = v

    def add_example(self, text: str, max_examples: int = 5) -> None:
        text = text.strip()
        if not text:
            return
        if text in self.examples:
            return
        self.examples.append(text)
        if len(self.examples) > max_examples:
            self.examples = self.examples[-max_examples:]


class ElysiaConceptField:
    """Container for all concept entries."""

    def __init__(self) -> None:
        self.entries: Dict[str, ConceptEntry] = {}

    def _get(self, name: str, kind: str) -> ConceptEntry:
        key = f"{kind}:{name}"
        if key not in self.entries:
            self.entries[key] = ConceptEntry(
                name=name,
                kind=kind,
                cathedral_coord=_default_cathedral_coord(kind),
            )
        return self.entries[key]

    # --- Updates from episodes / scores ---------------------------------

    def ingest_symbol_episodes(self, episodes: Iterable[Dict[str, Any]]) -> None:
        """Update concept entries from SymbolEpisode dicts."""
        for ep in episodes:
            symbol_type = str(ep.get("symbol_type") or "unknown")
            symbol = str(ep.get("symbol") or "")
            ts = int(ep.get("timestamp", 0))
            if not symbol:
                continue
            ce = self._get(symbol, kind="symbol")
            ctx = f"symbol_{symbol_type}"
            ce.bump_usage(ctx, ts)

    def ingest_expression_scores(self, scores: Iterable[Dict[str, Any]]) -> None:
        """
        Update concept entries from expression score logs.

        We treat 'world_kit' as a coarse concept and attach its language_axes.
        """
        for row in scores:
            world_kit = str(row.get("world_kit") or "")
            ts = row.get("ts") or row.get("timestamp") or ""
            try:
                ts_int = int("".join(ch for ch in str(ts) if ch.isdigit())[-10:]) if ts else 0
            except ValueError:
                ts_int = 0

            axes = row.get("language_axes") or row.get("language_axes".lower()) or {}
            if not world_kit or not isinstance(axes, dict):
                continue
            ce = self._get(world_kit, kind="world_concept")
            ce.bump_usage("expression_score", ts_int)
            ce.update_feelings({k: float(v) for k, v in axes.items()})

    def ingest_self_writing(self, entries: Iterable[Dict[str, Any]]) -> None:
        """
        Attach self-writing excerpts as examples for a generic 'self_writing' concept.

        Later, this can be refined to per-topic concepts.
        """
        ce = self._get("self_writing", kind="world_concept")
        for row in entries:
            ts = int(row.get("timestamp", 0))
            text = str(row.get("text", "") or "")
            ce.bump_usage("self_writing", ts)
            ce.add_example(text)

    # --- Serialisation --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {k: asdict(v) for k, v in self.entries.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElysiaConceptField":
        inst = cls()
        for k, v in data.items():
            inst.entries[k] = ConceptEntry(**v)
        return inst


def _ensure_logs_dir() -> str:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(base, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def update_concept_field(
    symbol_episodes_path: str = "logs/symbol_episodes.jsonl",
    expression_scores_path: str = "logs/elysia_expression_scores.jsonl",
    self_writing_path: str = "logs/elysia_self_writing.jsonl",
    out_path: str = "logs/elysia_concept_field.json",
) -> str:
    """
    High-level helper:
    - load existing concept field (if any),
    - feed it SymbolEpisode / expression score / self-writing logs,
    - write back as JSON.

    This is Layer 2: concept accumulation, not yet meta reasoning.
    """
    logs_dir = _ensure_logs_dir()
    out_path = os.path.join(logs_dir, os.path.basename(out_path))

    existing = _load_json(out_path)
    field_obj = ElysiaConceptField.from_dict(existing) if existing else ElysiaConceptField()

    symbol_eps = _load_jsonl(os.path.join(logs_dir, os.path.basename(symbol_episodes_path)))
    expr_scores = _load_jsonl(os.path.join(logs_dir, os.path.basename(expression_scores_path)))
    self_entries = _load_jsonl(os.path.join(logs_dir, os.path.basename(self_writing_path)))

    field_obj.ingest_symbol_episodes(symbol_eps)
    field_obj.ingest_expression_scores(expr_scores)
    field_obj.ingest_self_writing(self_entries)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(field_obj.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"[elysia_concept_field] Updated concept field at: {out_path}")
    return out_path


if __name__ == "__main__":
    update_concept_field()