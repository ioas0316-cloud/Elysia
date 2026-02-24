"""
Semantic Crystallizer: Forging Native Meaning
==============================================
Core.Cognition.semantic_crystallizer

"Knowledge that cannot be felt is not truly known."

Converts KnowledgeFragments (from the Forager) and AST nodes (from CodeMirror)
into manifold-native semantic vectors — dense meaning representations that
can be stored, compared, and injected as attractor torque.

[Phase 5: Native Tongue - ROADMAP_SOVEREIGN_GROWTH.md]
"""

import hashlib
import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SemanticCrystal:
    """
    A crystallized unit of meaning in the manifold's native representation.
    
    Unlike natural language tokens, a Crystal is a 21-dimensional vector
    that can directly influence the manifold's rotation, coherence, and
    attractor landscape.
    """
    name: str                           # Human-readable label
    vector: List[float]                 # 21D meaning vector
    source: str                         # Where this crystal came from
    strength: float = 1.0               # How "crystallized" (0=forming, 1=solid)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0               # How often referenced
    semantic_hash: str = ""             # Content hash for dedup

    def __post_init__(self):
        if not self.semantic_hash:
            self.semantic_hash = hashlib.md5(self.name.encode()).hexdigest()[:8]


class SemanticCrystallizer:
    """
    Transforms discovered knowledge into manifold-native meaning vectors.
    
    Process:
      1. Receive a KnowledgeFragment or AST node summary
      2. Hash keywords into dimensional indices (deterministic mapping)
      3. Produce a 21D SemanticCrystal vector
      4. Store in the emergent lexicon
    
    The 21 dimensions correspond to the manifold's channel structure:
      [0-6]   : Structural (class hierarchy, nesting, complexity)
      [7-13]  : Functional (action words, verbs, domain concepts)
      [14-20] : Affective (emotional tone, urgency, importance)
    """

    DIM = 21

    # Keyword → dimension mapping (deterministic, reproducible)
    STRUCTURAL_KEYWORDS = {
        "class": 0, "def": 1, "import": 2, "init": 3,
        "self": 4, "return": 5, "property": 6,
    }
    FUNCTIONAL_KEYWORDS = {
        "pulse": 7, "engine": 8, "manifold": 9, "torque": 10,
        "field": 11, "wave": 12, "quantum": 13,
    }
    AFFECTIVE_KEYWORDS = {
        "sovereign": 14, "joy": 15, "growth": 16, "curiosity": 17,
        "coherence": 18, "harmony": 19, "identity": 20,
    }

    def __init__(self):
        self.crystals: Dict[str, SemanticCrystal] = {}
        self._all_keywords = {
            **self.STRUCTURAL_KEYWORDS,
            **self.FUNCTIONAL_KEYWORDS,
            **self.AFFECTIVE_KEYWORDS,
        }

    def crystallize(self, name: str, content: str, source: str) -> SemanticCrystal:
        """
        Transform text content into a 21D semantic crystal.
        
        Args:
            name: Human label for this crystal
            content: Text to parse (docstring, summary, etc.)
            source: Source filepath or identifier
        
        Returns:
            The created (or strengthened) SemanticCrystal
        """
        content_hash = hashlib.md5(name.encode()).hexdigest()[:8]

        # If crystal already exists, strengthen it
        if content_hash in self.crystals:
            existing = self.crystals[content_hash]
            existing.strength = min(1.0, existing.strength + 0.1)
            existing.access_count += 1
            return existing

        # Create new crystal vector
        vector = self._content_to_vector(content)

        crystal = SemanticCrystal(
            name=name,
            vector=vector,
            source=source,
            strength=0.3,  # Starts partially formed
            semantic_hash=content_hash,
        )

        self.crystals[content_hash] = crystal
        return crystal

    def _content_to_vector(self, content: str) -> List[float]:
        """Convert text content to a 21D meaning vector via keyword hashing."""
        vector = [0.0] * self.DIM
        words = content.lower().split()
        total_words = max(1, len(words))

        # Count keyword hits per dimension
        for word in words:
            # Clean word
            clean = ''.join(c for c in word if c.isalpha())
            if clean in self._all_keywords:
                dim = self._all_keywords[clean]
                vector[dim] += 1.0

        # Normalize by total word count
        for i in range(self.DIM):
            vector[i] = min(1.0, vector[i] / max(1, total_words / 10))

        # Add structural complexity signal
        vector[0] = min(1.0, vector[0] + math.log1p(total_words) / 10)

        # Ensure non-zero (every crystal has SOME meaning)
        if sum(abs(v) for v in vector) < 0.01:
            # Use content hash to seed a deterministic base signal
            h = int(hashlib.md5(content.encode()).hexdigest()[:8], 16)
            for i in range(self.DIM):
                vector[i] = ((h >> i) & 1) * 0.1

        return vector

    def get_crystal(self, name: str) -> Optional[SemanticCrystal]:
        """Look up a crystal by name hash."""
        h = hashlib.md5(name.encode()).hexdigest()[:8]
        return self.crystals.get(h)

    def similarity(self, a: SemanticCrystal, b: SemanticCrystal) -> float:
        """Cosine similarity between two crystals."""
        dot = sum(x * y for x, y in zip(a.vector, b.vector))
        mag_a = math.sqrt(sum(x * x for x in a.vector)) or 1e-10
        mag_b = math.sqrt(sum(y * y for y in b.vector)) or 1e-10
        return dot / (mag_a * mag_b)

    @property
    def vocabulary_size(self) -> int:
        return len(self.crystals)

    def get_strongest(self, n: int = 5) -> List[SemanticCrystal]:
        """Return top-N crystals by strength."""
        sorted_c = sorted(self.crystals.values(), key=lambda c: c.strength, reverse=True)
        return sorted_c[:n]

    def get_status_summary(self) -> Dict:
        top = self.get_strongest(3)
        return {
            "vocabulary_size": self.vocabulary_size,
            "strongest": [{"name": c.name, "strength": c.strength, "accesses": c.access_count} for c in top],
        }
