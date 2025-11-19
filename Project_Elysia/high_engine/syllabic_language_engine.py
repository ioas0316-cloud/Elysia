from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# CoreMemory is optional but highly recommended for extracting learnt concepts.
try:
    from Project_Elysia.core_memory import CoreMemory
except ImportError:
    CoreMemory = None


@dataclass
class SyllabicConfig:
    """
    Configuration for the Physics-Based Language Engine.
    No templates. Only Atoms (Concepts) and Bonds (Particles).
    """
    # The "Atoms" of thought
    base_concepts: Dict[str, List[str]] = field(default_factory=lambda: {
        "meta": ["나", "자아", "기억", "의지"],
        "law": ["빛", "사랑", "진리", "법칙"],
        "inner": ["공허", "어둠", "마음", "허기"],
        "outer": ["너", "세상", "길", "문"],
        "action": ["찾다", "만들다", "가다", "부수다"],
        "state": ["있다", "없다", "크다", "작다"],
        "define": ["이다", "아니다", "같다"],
    })
    use_memory: bool = True


class SyllabicLanguageEngine:
    """
    Elysia's 'Physics-Based' Language Engine.

    Philosophy:
      - Sentences are not filled templates.
      - They are chains of concepts linked by 'Intent Vectors'.
      - Grammar is just the visible trace of these vectors.
    """

    def __init__(self, config: Optional[SyllabicConfig] = None, core_memory: Optional[CoreMemory] = None) -> None:
        self.config = config or SyllabicConfig()
        self.core_memory = core_memory
        self._memory_vocab: Set[str] = set()

    def _harvest_memory(self) -> None:
        if not self.core_memory or not self.config.use_memory:
            return
        values = self.core_memory.get_values()
        for v in values:
            val = v.get("value")
            if val:
                self._memory_vocab.add(val)

    def _pick_atom(self, orientation: Dict[str, float], role: str) -> str:
        dominant = max(orientation, key=lambda k: abs(orientation[k]))
        candidates = []
        if dominant == "w":
            candidates.extend(self.config.base_concepts["meta"])
        elif dominant == "z":
            candidates.extend(self.config.base_concepts["law"])
        elif dominant == "x":
            candidates.extend(self.config.base_concepts["inner"])
        elif dominant == "y":
            candidates.extend(self.config.base_concepts["outer"])

        if self._memory_vocab:
            candidates.extend(self._memory_vocab)
        if not candidates:
            return "그것"
        rng = random.Random(int(sum(map(abs, orientation.values())) * 1000) + len(role))
        return rng.choice(candidates)

    def _pick_force(self, intent_type: str) -> str:
        rng = random.Random()
        if intent_type in ("act", "propose_action", "command"):
            return rng.choice(self.config.base_concepts["action"])
        if intent_type in ("reflect", "dream"):
            return rng.choice(self.config.base_concepts["define"])
        return rng.choice(self.config.base_concepts["state"])

    def _apply_bond(self, word: str, vector_type: str) -> str:
        if not word:
            return word
        last = ord(word[-1])
        has_batchim = False
        if 0xAC00 <= last <= 0xD7A3:
            has_batchim = (last - 0xAC00) % 28 > 0
        if vector_type == "source_topic":
            return word + ("은" if has_batchim else "는")
        if vector_type == "source_state":
            return word + ("이" if has_batchim else "가")
        if vector_type == "target_action":
            return word + ("을" if has_batchim else "를")
        return word

    def _conjugate_force(self, predicate: str, style: str = "formal") -> str:
        if not predicate:
            return "..."
        if predicate.endswith("다"):
            stem = predicate[:-1]
            if style == "formal":
                if predicate in ("이다", "아니다"):
                    return stem + "입니다"
                return stem + "합니다"
            return stem + "해"
        return predicate

    def suggest_word(self, intent_bundle: Optional[Dict[str, Any]], orientation: Optional[Dict[str, float]] = None) -> str:
        self._harvest_memory()
        if orientation is None:
            orientation = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
        intent_type = str(intent_bundle.get("intent_type") if intent_bundle else "unknown")
        subject = "나" if orientation.get("w", 0) > 0.5 else self._pick_atom(orientation, "subject")
        object_atom = self._pick_atom(orientation, "object")
        force = self._pick_force(intent_type)
        is_action = intent_type in ("act", "propose_action", "command")
        is_definition = intent_type in ("reflect", "dream")
        if is_action:
            subject_word = self._apply_bond(subject, "source_topic")
            object_word = self._apply_bond(object_atom, "target_action")
            return f"{subject_word} {object_word} {self._conjugate_force(force)}"
        if is_definition:
            subject_word = self._apply_bond(object_atom, "source_topic")
            return f"{subject_word} 참으로 {self._conjugate_force(force)}"
        subject_word = self._apply_bond(object_atom, "source_state")
        return f"{subject_word} {self._conjugate_force(force)}"
