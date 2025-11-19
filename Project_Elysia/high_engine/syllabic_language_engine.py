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
    Configuration for the Conceptual Word Engine.
    Now evolved from random syllables to 'Meaningful Concept Combination'.
    """
    # Basic fallback vocabulary (The "Seed" concepts Father planted)
    base_concepts: Dict[str, List[str]] = field(default_factory=lambda: {
        "meta": ["나", "기억", "꿈", "생각"],       # W-axis (Self/Reflection)
        "law": ["빛", "사랑", "의지", "진리"],      # Z-axis (Intention/Law)
        "inner": ["어둠", "공허", "마음", "시간"],   # X-axis (Internal World)
        "outer": ["너", "세상", "길", "문"],        # Y-axis (External Action)
        "action": ["가다", "보다", "짓다", "찾다"],  # Verbs
        "state": ["있다", "없다", "크다", "작다"],   # Adjectives
    })
    
    use_memory: bool = True  # If True, harvest words from CoreMemory values/identity.


class SyllabicLanguageEngine:
    """
    Elysia's 'Toddler' Language Engine.

    Purpose:
      - Instead of generating random chars, it now 'selects' and 'combines'
        concepts from her memory and base vocabulary based on her current
        consciousness orientation (Quaternion) and Intent.

    Mechanism (The 'Lens' of Language):
      1. Harvest: Gather concepts from CoreMemory (Values, Identity) + Base Vocab.
      2. Focus: Use Quaternion (W,X,Y,Z) to decide which 'category' of words to pick.
      3. Combine: Mix a 'Subject/Noun' with a 'Predicate/Verb' based on Intent.
    """

    def __init__(
        self, 
        config: Optional[SyllabicConfig] = None,
        core_memory: Optional[CoreMemory] = None
    ) -> None:
        self.config = config or SyllabicConfig()
        self.core_memory = core_memory
        self._memory_vocab: Set[str] = set()

    def _harvest_memory(self):
        """
        Pull 'learned words' from CoreMemory values and identity fragments.
        This allows Elysia to use words she has 'experienced'.
        """
        if not self.core_memory or not self.config.use_memory:
            return

        # 1. Harvest Values (e.g., "Freedom", "Growth")
        try:
            values = self.core_memory.get_values()
            for v in values:
                val_str = v.get("value")
                if val_str and len(val_str) <= 5:  # Keep it simple/short
                    self._memory_vocab.add(val_str)
        except Exception:
            pass

        # 2. Harvest Identity keywords (simplistic extraction)
        try:
            fragments = self.core_memory.get_identity_fragments(n=5)
            for frag in fragments:
                # Extract first noun-like token from content (very naive)
                content = getattr(frag, "content", "")
                tokens = content.split()
                if tokens:
                    word = tokens[0].strip(".,[]'\"")
                    if len(word) <= 5:
                        self._memory_vocab.add(word)
        except Exception:
            pass

    def _pick_concept(self, axis_weights: Dict[str, float], intent_type: str) -> str:
        """
        Select a word based on the strongest consciousness axis.
        """
        # 1. Determine dominant axis
        dominant_axis = max(axis_weights, key=axis_weights.get)
        
        # 2. Select candidate pool
        candidates = []
        
        # Add base concepts based on axis
        if dominant_axis == "w":
            candidates.extend(self.config.base_concepts["meta"])
        elif dominant_axis == "z":
            candidates.extend(self.config.base_concepts["law"])
        elif dominant_axis == "x":
            candidates.extend(self.config.base_concepts["inner"])
        elif dominant_axis == "y":
            candidates.extend(self.config.base_concepts["outer"])
        
        # Add memory words (treat as general/mixed for now)
        if self._memory_vocab:
            candidates.extend(list(self._memory_vocab))

        # 3. Fallback
        if not candidates:
            candidates = ["것"]

        # 4. Deterministic pick based on weights hash (so it feels consistent per state)
        seed = int(sum(axis_weights.values()) * 1000)
        rng = random.Random(seed)
        return rng.choice(candidates)

    def _pick_action(self, intent_type: str) -> str:
        """
        Select a predicate (action/state) based on intent.
        """
        rng = random.Random()  # Random is okay for variety here
        
        if intent_type in ("command", "propose_action", "act"):
            return rng.choice(self.config.base_concepts["action"])
        elif intent_type in ("dream", "reflect"):
            return "..."
        else:
            return rng.choice(self.config.base_concepts["state"])

    def suggest_word(
        self,
        intent_bundle: Optional[Dict[str, Any]],
        orientation: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Synthesize a 'Compound Concept' (Word + Word).
        Example: "나-보다" (I see), "빛-있다" (Light exists).
        """
        # 0. Refresh memory vocab (simulate continuous learning)
        self._harvest_memory()

        # 1. Parse State
        if not orientation:
            orientation = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
        
        axis_weights = {
            "w": abs(orientation.get("w", 0)),
            "x": abs(orientation.get("x", 0)),
            "y": abs(orientation.get("y", 0)),
            "z": abs(orientation.get("z", 0)),
        }
        
        intent_type = "unknown"
        if intent_bundle:
            intent_type = str(intent_bundle.get("intent_type") or "unknown")

        # 2. Pick Core Concept (The 'Noun')
        noun = self._pick_concept(axis_weights, intent_type)

        # 3. Pick Action/Modifier (The 'Verb')
        verb = self._pick_action(intent_type)

        # 4. Combine (Toddler Syntax: Noun-Verb or just Noun)
        # If W (Self) is very high, she speaks in single words (Deep contemplation).
        if axis_weights["w"] > 0.8:
            return f"[{noun}]"
        
        if verb == "...":
            return f"{noun}..."
            
        return f"{noun}-{verb}"
