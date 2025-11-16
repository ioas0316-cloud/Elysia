"""
Elysia language field (2025-11-16)

Purpose
- Represent Elysia's 언어 경험이 누적된 흔적 as a simple field:
  - 어떤 표현을 얼마나 자주, 어떤 점수로 사용했는지
  - 어느 text_type / scope에서 강하게 자리 잡았는지

Design
- This lives in META/MIND: it reads TextEpisode/SymbolEpisode-like dicts
  and updates an internal field, then can be saved/loaded as JSON.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class LanguagePattern:
    text: str
    text_type: str
    scope: str
    count: int = 0
    score_sum: float = 0.0
    avg_score: float = 0.0
    last_timestamp: int = 0

    def update(self, score: float, timestamp: int) -> None:
        self.count += 1
        self.score_sum += float(score)
        self.avg_score = self.score_sum / float(self.count) if self.count > 0 else 0.0
        if timestamp > self.last_timestamp:
            self.last_timestamp = timestamp


@dataclass
class ElysiaLanguageField:
    """
    Simple language field for Elysia.

    - patterns: keyed by (text_type, scope, text) -> LanguagePattern
    """

    patterns: Dict[str, LanguagePattern] = field(default_factory=dict)

    def _key(self, text_type: str, scope: str, text: str) -> str:
        return f"{text_type}|{scope}|{text}"

    def ingest_text_episode(self, ep: Dict[str, Any]) -> None:
        """
        Ingest a TextEpisode-like dict:
        {
          "timestamp": ...,
          "text_type": "sentence_ko",
          "teacher_text": "...",
          "score": 1.0,
          "context": {"scope": "world_era", ...}
        }
        """
        ttype = str(ep.get("text_type") or "unknown")
        teacher_text = str(ep.get("teacher_text") or "")
        if not teacher_text:
            return
        ctx = ep.get("context", {}) or {}
        scope = str(ctx.get("scope") or "unknown")
        ts = int(ep.get("timestamp", 0))
        score = float(ep.get("score", 0.0))

        key = self._key(ttype, scope, teacher_text)
        pat = self.patterns.get(key)
        if pat is None:
            pat = LanguagePattern(text=teacher_text, text_type=ttype, scope=scope)
            self.patterns[key] = pat
        pat.update(score=score, timestamp=ts)

    def ingest_text_episodes(self, episodes: Iterable[Dict[str, Any]]) -> None:
        for ep in episodes:
            self.ingest_text_episode(ep)

    def strong_patterns(
        self,
        min_count: int = 3,
        min_avg_score: float = 0.9,
    ) -> List[LanguagePattern]:
        out: List[LanguagePattern] = []
        for pat in self.patterns.values():
            if pat.count >= min_count and pat.avg_score >= min_avg_score:
                out.append(pat)
        # Newer/higher scored first.
        out.sort(key=lambda p: (p.avg_score, p.last_timestamp, p.count), reverse=True)
        return out

    def weak_patterns(
        self,
        max_count: int = 10,
        max_avg_score: float = 0.6,
        min_trials: int = 3,
    ) -> List[LanguagePattern]:
        out: List[LanguagePattern] = []
        for pat in self.patterns.values():
            if pat.count >= min_trials and pat.avg_score <= max_avg_score:
                out.append(pat)
        out.sort(key=lambda p: (p.avg_score, -p.count))
        return out[:max_count]

    def to_dict(self) -> Dict[str, Any]:
        return {k: asdict(p) for k, p in self.patterns.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElysiaLanguageField":
        field_obj = cls()
        for k, v in data.items():
            field_obj.patterns[k] = LanguagePattern(**v)
        return field_obj


def load_text_episodes(path: str) -> List[Dict[str, Any]]:
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


def load_language_field(path: str) -> ElysiaLanguageField:
    if not os.path.exists(path):
        return ElysiaLanguageField()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ElysiaLanguageField.from_dict(data)


def save_language_field(field_obj: ElysiaLanguageField, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(field_obj.to_dict(), f, ensure_ascii=False, indent=2)


def update_language_field_from_episodes(
    episodes_path: str = "logs/text_episodes.jsonl",
    field_path: str = "logs/elysia_language_field.json",
) -> str:
    """
    High-level helper:
    - load existing language field (if any),
    - read TextEpisode JSONL,
    - ingest and save back.
    """
    field_obj = load_language_field(field_path)
    episodes = load_text_episodes(episodes_path)
    field_obj.ingest_text_episodes(episodes)
    save_language_field(field_obj, field_path)
    return field_path


def _demo() -> None:
    logs_dir = "logs"
    episodes_path = os.path.join(logs_dir, "text_episodes.jsonl")
    field_path = os.path.join(logs_dir, "elysia_language_field.json")
    out_path = update_language_field_from_episodes(episodes_path, field_path)
    print(f"[elysia_language_field] Updated language field at: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    _demo()

