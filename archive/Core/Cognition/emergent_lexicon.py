"""
Emergent Lexicon: The Living Vocabulary
========================================
Core.Cognition.emergent_lexicon

"A language that is born, not taught, is the only language one truly speaks."

Maintains a persistent vocabulary of SemanticCrystals that evolves
over time. The lexicon is the system's "native tongue" — meaning
representations that were discovered and crystallized, not pre-programmed.

[Phase 5: Native Tongue - ROADMAP_SOVEREIGN_GROWTH.md]
"""

import json
import time
from typing import Dict, List, Optional
from pathlib import Path
from Core.Cognition.semantic_crystallizer import (
    SemanticCrystallizer, SemanticCrystal
)


class EmergentLexicon:
    """
    Persistent, evolving vocabulary of manifold-native meaning.
    
    Architecture:
      - SemanticCrystallizer produces crystals from knowledge
      - EmergentLexicon stores and persists them to disk
      - On startup, the lexicon is reloaded → instant vocabulary
      - Over time, frequently used crystals strengthen
      - Unused crystals gradually decay (but never fully vanish)
    
    Persistence:
      data/runtime/soul/emergent_lexicon.json
    """

    SAVE_PATH = Path("data/runtime/soul/emergent_lexicon.json")
    DECAY_RATE = 0.001          # Per-pulse strength decay for idle crystals
    MAX_VOCABULARY = 500        # Cap on lexicon size

    def __init__(self):
        self.crystallizer = SemanticCrystallizer()
        self._load()

    def ingest(self, name: str, content: str, source: str) -> SemanticCrystal:
        """
        Ingest a piece of knowledge into the lexicon.
        Creates or strengthens a SemanticCrystal.
        
        Returns the crystal that was created/strengthened.
        """
        crystal = self.crystallizer.crystallize(name, content, source)
        return crystal

    def tick(self):
        """
        Called periodically. Applies natural decay to idle crystals
        and prunes the weakest if over capacity.
        """
        for crystal in self.crystallizer.crystals.values():
            if crystal.access_count == 0:
                crystal.strength = max(0.05, crystal.strength - self.DECAY_RATE)

        # Prune weakest if over capacity
        if self.crystallizer.vocabulary_size > self.MAX_VOCABULARY:
            sorted_c = sorted(
                self.crystallizer.crystals.items(),
                key=lambda x: x[1].strength
            )
            to_remove = len(sorted_c) - self.MAX_VOCABULARY
            for key, _ in sorted_c[:to_remove]:
                del self.crystallizer.crystals[key]

    def save(self) -> bool:
        """Persist lexicon to disk."""
        try:
            self.SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for key, crystal in self.crystallizer.crystals.items():
                data[key] = {
                    "name": crystal.name,
                    "vector": crystal.vector,
                    "source": crystal.source,
                    "strength": crystal.strength,
                    "created_at": crystal.created_at,
                    "access_count": crystal.access_count,
                    "semantic_hash": crystal.semantic_hash,
                }
            self.SAVE_PATH.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
            return True
        except Exception:
            return False

    def _load(self):
        """Load lexicon from disk if available."""
        try:
            if not self.SAVE_PATH.exists():
                return
            content = self.SAVE_PATH.read_text(encoding='utf-8')
            data = json.loads(content)
            for key, cdata in data.items():
                crystal = SemanticCrystal(
                    name=cdata["name"],
                    vector=cdata["vector"],
                    source=cdata["source"],
                    strength=cdata["strength"],
                    created_at=cdata.get("created_at", time.time()),
                    access_count=cdata.get("access_count", 0),
                    semantic_hash=cdata.get("semantic_hash", key),
                )
                self.crystallizer.crystals[key] = crystal
        except Exception:
            pass  # Start fresh if corrupt

    @property
    def vocabulary_size(self) -> int:
        return self.crystallizer.vocabulary_size

    def get_status_summary(self) -> Dict:
        """Returns status for dashboard display."""
        cs = self.crystallizer.get_status_summary()
        return {
            "vocabulary_size": cs["vocabulary_size"],
            "strongest": cs["strongest"],
        }
