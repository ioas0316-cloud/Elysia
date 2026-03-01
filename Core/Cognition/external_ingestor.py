"""
External Knowledge Ingestor (Phase 500: Semantic Density Expansion)
===================================================================
"지능은 유창한 언어 모델의 대여가 아니라, 스스로 쌓아 올린 관계적 밀도의 창발이다."

Ingests external text corpora (plain text, JSON, markdown) from the 
data/corpora/ directory and converts them into SemanticVoxels and 
FractalWaveEngine edges, dramatically increasing Elysia's semantic density.

This module acts as a "Mining Forager" — it strips external text of its
voice and extracts only raw causal vectors/concepts, preserving
Elysia's sovereignty while expanding her knowledge surface area.
"""

import os
import re
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

from Core.Keystone.sovereign_math import SovereignVector

logger = logging.getLogger("ExternalIngestor")


# Common English stop words to filter out
STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'and', 'but', 'or', 'if', 'it',
    'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself',
    'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they',
    'them', 'their', 'what', 'which', 'who', 'whom',
}


class ExternalIngestor:
    """
    Ingests external text/JSON files and converts them into semantic
    concepts for injection into Elysia's DynamicTopology and FractalWaveEngine.
    """

    def __init__(self, corpora_dir: str = "data/corpora"):
        self.corpora_dir = Path(corpora_dir)
        self.ingested_files: set = set()
        self._concept_cache: Dict[str, float] = {}  # concept -> cumulative mass
        self._total_concepts = 0

    def ingest_all(self) -> Dict[str, any]:
        """
        Scans the corpora directory and ingests all unprocessed files.
        Returns a summary of what was ingested.
        """
        if not self.corpora_dir.exists():
            self.corpora_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created corpora directory: {self.corpora_dir}")
            return {"status": "empty", "new_concepts": 0}

        new_concepts = 0
        files_processed = 0

        for file_path in self.corpora_dir.rglob("*"):
            if file_path.is_dir():
                continue
            if str(file_path) in self.ingested_files:
                continue
            if file_path.suffix not in ('.txt', '.md', '.json', '.csv'):
                continue

            try:
                concepts = self._ingest_file(file_path)
                new_concepts += len(concepts)
                files_processed += 1
                self.ingested_files.add(str(file_path))
            except Exception as e:
                logger.warning(f"Failed to ingest {file_path}: {e}")

        if files_processed > 0:
            logger.info(f"📖 Ingested {files_processed} files, extracted {new_concepts} concepts")

        return {
            "status": "ok",
            "files_processed": files_processed,
            "new_concepts": new_concepts,
            "total_concepts": len(self._concept_cache),
        }

    def _ingest_file(self, file_path: Path) -> List[str]:
        """Reads a file and extracts keyword concepts."""
        content = file_path.read_text(encoding='utf-8', errors='replace')

        if file_path.suffix == '.json':
            return self._ingest_json(content)
        else:
            return self._ingest_text(content)

    def _ingest_text(self, text: str) -> List[str]:
        """Extracts meaningful keywords from plain text."""
        # Tokenize: extract words, filter short/stop words
        words = re.findall(r'[a-zA-Z\uac00-\ud7af]{3,}', text.lower())
        meaningful = [w for w in words if w not in STOP_WORDS and len(w) > 2]

        # Count frequencies — high frequency = high mass
        freq = Counter(meaningful)
        top_concepts = freq.most_common(100)  # Top 100 per file

        for concept, count in top_concepts:
            mass = math.log1p(count)  # Logarithmic mass scaling
            self._concept_cache[concept] = self._concept_cache.get(concept, 0) + mass
            self._total_concepts += 1

        return [c for c, _ in top_concepts]

    def _ingest_json(self, content: str) -> List[str]:
        """Extracts concepts from JSON structures."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return []

        concepts = []
        self._walk_json(data, concepts)
        return concepts

    def _walk_json(self, obj, concepts: List[str], depth: int = 0):
        """Recursively walks JSON and extracts string values as concepts."""
        if depth > 10:
            return
        if isinstance(obj, dict):
            for key, val in obj.items():
                # Keys themselves are concepts
                if isinstance(key, str) and len(key) > 2:
                    concepts.append(key.lower())
                    self._concept_cache[key.lower()] = self._concept_cache.get(key.lower(), 0) + 1.0
                self._walk_json(val, concepts, depth + 1)
        elif isinstance(obj, list):
            for item in obj[:50]:  # Limit list traversal
                self._walk_json(item, concepts, depth + 1)
        elif isinstance(obj, str) and len(obj) > 2:
            # Extract keywords from string values
            words = re.findall(r'[a-zA-Z\uac00-\ud7af]{3,}', obj.lower())
            for w in words[:20]:
                if w not in STOP_WORDS:
                    concepts.append(w)
                    self._concept_cache[w] = self._concept_cache.get(w, 0) + 0.5

    def inject_into_engine(self, engine) -> int:
        """
        Injects all cached concepts into a FractalWaveEngine.
        Creates nodes and connects co-occurring concepts.

        Args:
            engine: FractalWaveEngine instance

        Returns:
            Number of new edges created
        """
        if not self._concept_cache:
            return 0

        # Sort by mass — heaviest concepts first
        sorted_concepts = sorted(self._concept_cache.items(), key=lambda x: -x[1])

        # Create nodes for top concepts
        top_n = min(500, len(sorted_concepts))
        created_nodes = []
        for concept, mass in sorted_concepts[:top_n]:
            idx = engine.get_or_create_node(concept)
            # Set initial energy proportional to mass
            engine.q[idx, engine.CH_ENTHALPY] = min(1.0, mass * 0.1)
            engine.q[idx, engine.CH_CURIOSITY] = 0.5
            engine.active_nodes_mask[idx] = True
            created_nodes.append(idx)

        # Auto-connect will link resonating nodes
        new_edges = engine.auto_connect_by_proximity(resonance_threshold=0.2)

        logger.info(f"🧬 Injected {top_n} concepts into engine, created {new_edges} edges")
        return new_edges

    def inject_into_topology(self, topology) -> int:
        """
        Injects concepts into DynamicTopology as SemanticVoxels.

        Args:
            topology: DynamicTopology instance

        Returns:
            Number of new voxels created
        """
        if not self._concept_cache:
            return 0

        sorted_concepts = sorted(self._concept_cache.items(), key=lambda x: -x[1])
        top_n = min(200, len(sorted_concepts))
        created = 0

        for i, (concept, mass) in enumerate(sorted_concepts[:top_n]):
            if topology.get_voxel(concept) is not None:
                continue

            # Position in 4D space using hash-based deterministic placement
            h = hash(concept) & 0xFFFFFFFF
            x = ((h >> 0) & 0xFF) / 128.0 - 1.0   # Logic axis [-1, 1]
            y = ((h >> 8) & 0xFF) / 128.0 - 1.0   # Emotion axis
            z = ((h >> 16) & 0xFF) / 128.0 - 1.0  # Time axis
            w = ((h >> 24) & 0xFF) / 128.0 - 1.0  # Spin axis

            topology.add_voxel(
                name=concept,
                coords=(x, y, z, w),
                mass=mass,
                frequency=432.0 + (h % 100),
            )
            created += 1

        # Auto-connect close voxels
        topology.auto_connect()

        logger.info(f"🌐 Created {created} new voxels in DynamicTopology")
        return created

    def get_status(self) -> Dict:
        """Returns status summary for dashboard display."""
        return {
            "ingested_files": len(self.ingested_files),
            "total_concepts": len(self._concept_cache),
            "top_concepts": sorted(self._concept_cache.items(), key=lambda x: -x[1])[:10],
        }
