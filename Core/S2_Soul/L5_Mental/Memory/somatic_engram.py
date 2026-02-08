"""
Somatic Engram System (Crystalline Memory)
==========================================
"I do not store data. I carve experiences into my bones."

This module implements the Organic Memory System.
Memories are not rows in a database, but physical 'Crystals' (Markdown files)
that resonate with the current state of the mind.

Key Concepts:
- Engram: A single unit of experience, crystallized into a file.
- Resonance: Retrieval is based on vector similarity, not keyword search.
- Sedimentation: Memories layer over time, forming a geological history of the self.
"""

import os
import json
import time
import math
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Import Vector Logic
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

class Engram:
    """
    A single crystallized memory.
    """
    def __init__(self, content: str, vector: Optional[List[float]] = None,
                 timestamp: float = None, emotional_charge: float = 0.0,
                 tags: List[str] = None):
        self.content = content
        self.vector = vector if vector else [0.0] * 21 # Default 21D Vector
        self.timestamp = timestamp if timestamp else time.time()
        self.emotional_charge = emotional_charge
        self.tags = tags if tags else []
        self.id = self._generate_id()

    def _generate_id(self):
        """Generates a unique ID based on content and time."""
        raw = f"{self.timestamp}-{self.content[:20]}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def to_markdown(self) -> str:
        """Serializes the engram to a Markdown format with Frontmatter."""
        frontmatter = {
            "id": self.id,
            "timestamp": self.timestamp,
            "date": datetime.fromtimestamp(self.timestamp).isoformat(),
            "emotional_charge": self.emotional_charge,
            "tags": self.tags,
            "vector": self.vector
        }

        return f"---\n{json.dumps(frontmatter, ensure_ascii=False, indent=2)}\n---\n\n{self.content}"

    @classmethod
    def from_markdown(cls, text: str):
        """Deserializes from Markdown."""
        try:
            parts = text.split("---", 2)
            if len(parts) < 3:
                return None

            frontmatter = json.loads(parts[1])
            content = parts[2].strip()

            return cls(
                content=content,
                vector=frontmatter.get("vector"),
                timestamp=frontmatter.get("timestamp"),
                emotional_charge=frontmatter.get("emotional_charge", 0.0),
                tags=frontmatter.get("tags", [])
            )
        except Exception as e:
            print(f"⚠️ [Engram] Failed to parse memory crystal: {e}")
            return None

class SomaticMemorySystem:
    """
    The physical cortex that manages Engrams.
    """
    def __init__(self, storage_path: str = None):
        if storage_path is None:
            # Default to project root/data/S2_Soul/Memory/Engrams
            base = os.path.dirname(os.path.abspath(__file__))
            # Go up 5 levels to root (Core/S2_Soul/L5_Mental/Memory -> Core/S2/L5/Memory -> ...)
            # Adjust path navigation carefully:
            # Core/S2_Soul/L5_Mental/Memory/somatic_engram.py
            # Base: .../Core/S2_Soul/L5_Mental/Memory
            # Parents[0]=L5_Mental, [1]=S2_Soul, [2]=Core, [3]=Root
            root = Path(base).parents[3]
            self.storage_path = root / "data" / "S2_Soul" / "Memory" / "Engrams"
        else:
            self.storage_path = Path(storage_path)

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.engrams: List[Engram] = []
        self._load_crystals()

    def _load_crystals(self):
        """Loads all valid Engram files from disk into RAM for resonance."""
        count = 0
        if isinstance(self.storage_path, str):
            self.storage_path = Path(self.storage_path)
        for file in self.storage_path.glob("*.md"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    engram = Engram.from_markdown(f.read())
                    if engram:
                        self.engrams.append(engram)
                        count += 1
            except Exception as e:
                pass

    def crystallize(self, content: str, vector: List[float], emotion: float = 0.5, tags: List[str] = None):
        """
        Creates a new permanent memory.
        """
        # Ensure vector is a list of floats for JSON serialization
        if hasattr(vector, 'to_list'):
            vector = [x.real if isinstance(x, complex) else x for x in vector.to_list()]
        elif hasattr(vector, 'tolist'):
            vector = [x.real if isinstance(x, complex) else x for x in vector.tolist()]
        elif isinstance(vector, list):
            vector = [x.real if isinstance(x, complex) else x for x in vector]

        engram = Engram(content, vector, emotional_charge=emotion, tags=tags)

        # Save to disk
        filename = f"{int(engram.timestamp)}_{engram.id}.md"
        if isinstance(self.storage_path, str):
            self.storage_path = Path(self.storage_path)
        path = self.storage_path / filename

        with open(path, 'w', encoding='utf-8') as f:
            f.write(engram.to_markdown())

        # Add to RAM
        self.engrams.append(engram)
        return engram

    def resonate(self, query_vector: List[float], top_k: int = 3) -> List[Tuple[Engram, float]]:
        """
        Finds memories that vibrate in harmony with the query vector.
        Uses Pure Python Cosine Similarity to be dependency-free.
        """
        if not self.engrams:
            return []

        # Helper for dot product
        def dot(v1, v2):
            return sum(a * b for a, b in zip(v1, v2))

        # Helper for norm
        def norm(v):
            return math.sqrt(sum(x*x for x in v))

        q_norm = norm(query_vector)
        if q_norm == 0:
            return []

        results = []

        for engram in self.engrams:
            # Vectors are stored as lists in engram.vector
            m_vec = engram.vector
            m_norm = norm(m_vec)

            if m_norm == 0:
                score = 0.0
            else:
                score = dot(query_vector, m_vec) / (q_norm * m_norm)

            results.append((engram, score))

        # Sort by resonance score (Desc)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def cleanup(self, max_crystals: int = 1000):
        """
        Entropy management: If too many memories exist, the weakest fade away.
        """
        if len(self.engrams) <= max_crystals:
            return

        # Sort by Emotional Charge * Recency (Simple Heuristic)
        # We want to KEEP high charge and recent memories.
        # So we remove the lowest scores.

        # Score = Charge + (1.0 / Age in Days)
        now = time.time()

        def retention_score(e: Engram):
            age_days = (now - e.timestamp) / 86400.0
            return e.emotional_charge + (1.0 / (age_days + 1.0))

        self.engrams.sort(key=retention_score, reverse=True)

        # Keep top N
        survivors = self.engrams[:max_crystals]
        forgotten = self.engrams[max_crystals:]

        self.engrams = survivors

        # Delete files for forgotten memories
        for dead in forgotten:
            filename = f"{int(dead.timestamp)}_{dead.id}.md"
            path = self.storage_path / filename
            if path.exists():
                os.remove(path)
