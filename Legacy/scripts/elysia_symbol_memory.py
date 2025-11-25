"""
Elysia-side symbol memory (2025-11-16).

Purpose
- Provide a tiny, explicit "기호 기억" 구조 for Elysia:
  - 어떤 symbol_type/symbol이 얼마나 자주, 얼마나 정확하게 사용됐는지
  - (후속 단계에서) WORLD 객체/장면과 어떤 식으로 함께 등장했는지

Design
- This is a META/MIND helper: it reads SymbolEpisode JSONL and maintains
  an internal association table, but never touches WORLD physics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class SymbolStats:
    count: int = 0
    correct: int = 0
    accuracy: float = 0.0
    last_timestamp: int = 0


@dataclass
class ElysiaSymbolMemory:
    """
    Simple symbol memory for Elysia.

    - by_type[symbol_type][symbol] -> SymbolStats
    """

    by_type: Dict[str, Dict[str, SymbolStats]] = field(default_factory=dict)

    def ingest_episode(self, episode: Dict[str, Any]) -> None:
        stype = str(episode.get("symbol_type") or "unknown")
        symbol = str(episode.get("symbol") or "")
        correct = bool(episode.get("correct", False))
        ts = int(episode.get("timestamp", 0))

        bucket = self.by_type.setdefault(stype, {})
        stats = bucket.get(symbol)
        if stats is None:
            stats = SymbolStats()
            bucket[symbol] = stats

        stats.count += 1
        if correct:
            stats.correct += 1
        stats.last_timestamp = max(stats.last_timestamp, ts)
        stats.accuracy = (stats.correct / stats.count) if stats.count > 0 else 0.0

    def ingest_episodes(self, episodes: Iterable[Dict[str, Any]]) -> None:
        for ep in episodes:
            self.ingest_episode(ep)

    def known_symbols(
        self,
        symbol_type: str,
        min_accuracy: float = 0.7,
        min_count: int = 5,
    ) -> List[str]:
        """
        Return symbols that Elysia can treat as "대체로 익힌" 기호.
        """
        bucket = self.by_type.get(symbol_type, {})
        out: List[str] = []
        for sym, stats in bucket.items():
            if stats.count >= min_count and stats.accuracy >= min_accuracy:
                out.append(sym)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            stype: {sym: asdict(stats) for sym, stats in bucket.items()}
            for stype, bucket in self.by_type.items()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElysiaSymbolMemory":
        mem = cls()
        for stype, bucket in data.items():
            typed_bucket: Dict[str, SymbolStats] = {}
            for sym, stats_dict in bucket.items():
                typed_bucket[sym] = SymbolStats(**stats_dict)
            mem.by_type[stype] = typed_bucket
        return mem


def load_symbol_episodes(path: str) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return episodes
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return episodes


def _demo() -> None:
    """
    Demo: load symbol_episodes.jsonl and print what Elysia "knows".
    """
    path = os.path.join("logs", "symbol_episodes.jsonl")
    episodes = load_symbol_episodes(path)
    if not episodes:
        print(f"[elysia_symbol_memory] No symbol episodes at {path}")
        return

    mem = ElysiaSymbolMemory()
    mem.ingest_episodes(episodes)

    for stype in ("letter_ko", "syllable_ko", "letter_en", "word_en"):
        known = mem.known_symbols(stype, min_accuracy=0.6, min_count=5)
        print(f"- {stype}: known={known}")


if __name__ == "__main__":
    _demo()

