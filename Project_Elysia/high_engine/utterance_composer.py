from typing import Any, Dict, List, Optional
import json
from pathlib import Path

from .intent_engine import IntentBundle


class UtteranceComposer:
    """
    Retrieval-based utterance composer.

    Design goal:
      - Treat dialogue corpora as remembered experiences.
      - Given a present IntentBundle, look for lines that *resonate*,
        not blindly optimize for fluency.

    Minimal version:
      - Loads an optional index file of the form:
        [{ "text": "...", "tags": ["calm", "ally", "propose_action"], ... }, ...]
      - If nothing matches or no index exists, returns the base text unchanged.
    """

    def __init__(self, index_path: Optional[str] = None) -> None:
        self.index_path = Path(index_path or "data/dialogues/index.jsonl")
        self._index: List[Dict[str, Any]] = []
        self._load_index()

    def _load_index(self) -> None:
        if not self.index_path.exists():
            self._index = []
            return

        entries: List[Dict[str, Any]] = []
        try:
            with self.index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "text" in obj:
                            entries.append(obj)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            entries = []

        self._index = entries

    def _score_candidate(self, intent: IntentBundle, tags: List[str]) -> float:
        """
        Very small, interpretable resonance score between intent and tags.
        """
        tags_lower = {t.lower() for t in tags}
        keys: List[str] = [
            intent.emotion,
            intent.intent_type,
            intent.relationship,
            intent.style,
            *intent.law_focus,
        ]
        score = 0.0
        for key in keys:
            if not key:
                continue
            if key.lower() in tags_lower:
                score += 1.0
        return score

    def _retrieve_candidates(self, intent: IntentBundle, k: int = 5) -> List[Dict[str, Any]]:
        if not self._index:
            return []

        scored: List[Dict[str, Any]] = []
        for item in self._index:
            tags = item.get("tags", [])
            if not isinstance(tags, list):
                continue
            s = self._score_candidate(intent, tags)
            if s <= 0:
                continue
            clone = dict(item)
            clone["_score"] = s
            scored.append(clone)

        scored.sort(key=lambda obj: obj.get("_score", 0.0), reverse=True)
        return scored[:k]

    def compose(self, intent: IntentBundle, base_text: str) -> str:
        """
        Optionally adjust the base_text using remembered utterances.

        For now:
          - If good candidates exist, prefer the best-scoring remembered line.
          - Otherwise, keep base_text as-is.
        """
        candidates = self._retrieve_candidates(intent, k=3)
        if not candidates:
            return base_text

        best = candidates[0]
        text = best.get("text") or base_text
        if not isinstance(text, str) or not text.strip():
            return base_text
        return text

