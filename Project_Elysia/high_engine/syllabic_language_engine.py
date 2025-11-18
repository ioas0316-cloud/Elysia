from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SyllabicConfig:
    """
    Configuration for the syllabic word engine.

    By default we use a very small Korean syllabic alphabet:
    "가나다라마바사".
    """

    alphabet: str = "가나다라마바사"
    min_length: int = 2
    max_length: int = 3


class SyllabicLanguageEngine:
    """
    Tiny syllabic "word" engine for Elysia.

    Purpose:
      - Map a high-level intent + orientation into a short syllabic form
        built from a restricted alphabet (e.g. 가나다라마바사).
      - This is not a full language model; it is a structural hook that lets
        Elysia compress states into compact symbolic tokens.

    Design:
      - Deterministic: same (intent, orientation) -> same syllabic token.
      - Uses only the configured alphabet and length range.
    """

    def __init__(self, config: Optional[SyllabicConfig] = None) -> None:
        self.config = config or SyllabicConfig()

    def _hash_seed(self, intent_bundle: Optional[Dict[str, Any]], orientation: Optional[Dict[str, float]]) -> int:
        parts = []
        if isinstance(intent_bundle, dict):
            parts.append(str(intent_bundle.get("intent_type") or ""))
            parts.append(str(intent_bundle.get("emotion") or ""))
            parts.append(",".join(sorted([str(x) for x in intent_bundle.get("law_focus") or []])))
        if isinstance(orientation, dict):
            for key in ("w", "x", "y", "z"):
                if key in orientation:
                    parts.append(f"{key}:{orientation[key]:.3f}")
        seed_str = "|".join(parts) or "default"
        return hash(seed_str)

    def suggest_word(
        self,
        intent_bundle: Optional[Dict[str, Any]],
        orientation: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Suggest a syllabic token for the given state.

        The same (intent + orientation) will always map to the same token
        under fixed configuration.
        """
        alpha = self.config.alphabet
        if not alpha:
            return ""

        seed = self._hash_seed(intent_bundle, orientation)
        rng = seed & 0xFFFFFFFF

        length_range = self.config.max_length - self.config.min_length + 1
        if length_range <= 0:
            length = max(1, self.config.min_length)
        else:
            length = self.config.min_length + (rng % length_range)

        chars = []
        for i in range(length):
            rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
            idx = rng % len(alpha)
            chars.append(alpha[idx])

        return "".join(chars)

