# [Genesis: 2025-12-02] Purified by Elysia
from __future__ import annotations

"""
Primordial language engine for the Law simulation.

This module treats waveform-style "proto words" as an emergent language channel
grounded in:
- a stimulus target (e.g. fire, water, wolf, wind),
- the actor's emotional state,
- and a coarse memory signal.

Each observation of a produced word is fed back into a lightweight lexicon that
tracks how well that form has "stuck" (using memory as a proxy signal).

On subsequent uses, the engine can bias toward the most stable form for a
given (stimulus, emotion) pair, acting as a simple self-correction loop
without introducing any external supervision.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Any
import json


@dataclass
class WordStats:
    """Running statistics for a single proto-word form."""

    count: int = 0
    total_memory: float = 0.0

    @property
    def avg_memory(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.total_memory / float(self.count)


class PrimordialLanguageEngine:
    """
    Tracks proto-word variants and gently corrects them toward stable forms.

    The engine does not invent semantics on its own; instead it:
    - receives a stimulus description (must contain "target"),
    - an emotion label,
    - and a numeric memory_strength signal,
    - proposes a proto-word (base + repeated suffix),
    - and records how often that form appears and under what memory levels.

    Over time, for each (base, emotion) pair, the engine will favor the
    highest-memory, most-used word form as the "canonical" variant.
    """

    def __init__(self, suffix_map: Dict[str, str], max_depth: int = 4):
        self.suffix_map: Dict[str, str] = dict(suffix_map)
        self.max_depth = int(max_depth)
        # key: (base, emotion) -> value: {word -> WordStats}
        self.lexicon: Dict[Tuple[str, str], Dict[str, WordStats]] = {}

    @staticmethod
    def _key(base: str, emotion: str) -> Tuple[str, str]:
        return str(base), str(emotion)

    def _fallback_word(self, base: str, emotion: str, memory_strength: float) -> str:
        """Deterministic fallback: base + suffix repeated by memory depth."""
        suffix = self.suffix_map.get(emotion, "ra")
        try:
            mem_val = float(memory_strength)
        except (TypeError, ValueError):
            mem_val = 0.0
        # Map roughly 0..100 memory into 1..max_depth syllable depth.
        depth = 1 + int(max(0.0, min(mem_val, 99.9)) // max(1.0, 100.0 / max(self.max_depth, 1)))
        depth = max(1, min(self.max_depth, depth))
        return f"{base} {suffix * depth}"

    def suggest_word(self, stimulus: Dict[str, Any], emotion: str, memory_strength: float) -> str:
        """
        Suggest a proto-word for the given context.

        If the engine has seen stable forms for this (base, emotion) pair, it
        will favor the form with the highest average memory (with a small
        preference for more frequently used forms). Otherwise it falls back to
        the raw waveform generation.
        """
        base = str(stimulus.get("target", "unknown"))
        key = self._key(base, emotion)
        variants = self.lexicon.get(key)

        if not variants:
            return self._fallback_word(base, emotion, memory_strength)

        best_word = None
        best_score = float("-inf")
        try:
            mem_now = float(memory_strength)
        except (TypeError, ValueError):
            mem_now = 0.0

        for word, stats in variants.items():
            # Score encourages high average memory and soft alignment with
            # the current memory context, plus a tiny count bonus to break ties.
            avg = stats.avg_memory
            alignment = -abs(avg - mem_now)
            score = avg + 0.1 * alignment + 0.01 * stats.count
            if score > best_score:
                best_score = score
                best_word = word

        if best_word is not None:
            return best_word
        return self._fallback_word(base, emotion, memory_strength)

    def observe(self, stimulus: Dict[str, Any], emotion: str, word: str, memory_strength: float) -> None:
        """
        Record that a given proto-word was actually used in context.

        This updates the running statistics that future suggestions will use
        to bias toward more stable, memorable forms.
        """
        base = str(stimulus.get("target", "unknown"))
        key = self._key(base, emotion)
        variants = self.lexicon.setdefault(key, {})
        stats = variants.get(word)
        if stats is None:
            stats = WordStats()
            variants[word] = stats
        stats.count += 1
        try:
            stats.total_memory += max(0.0, float(memory_strength))
        except (TypeError, ValueError):
            # If memory is not a valid float, ignore it but keep the count.
            pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the lexicon into a plain dict for logging or persistence."""
        data: Dict[str, Any] = {}
        for (base, emotion), variants in self.lexicon.items():
            key_str = f"{base}::{emotion}"
            data[key_str] = {
                word: asdict(stats)
                for word, stats in variants.items()
            }
        return data

    def save_json(self, path: Path) -> None:
        """Persist the current lexicon to a JSON file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            # Persistence is best-effort only; failures should not break the simulation.
            pass
