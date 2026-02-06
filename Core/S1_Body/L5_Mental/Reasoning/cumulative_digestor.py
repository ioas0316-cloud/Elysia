"""
Cumulative Digestor (지식 침전 엔진)
=====================================
Core.S1_Body.L5_Mental.Reasoning.cumulative_digestor

This module is the 'Stomach' of Elysia's knowledge.
It systematically digests the project's documentation, code comments, and 
user dialogues to ensure that vocabulary and concepts are persistently 
sedimented into the LightUniverse.
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

from Core.S1_Body.L6_Structure.Wave.light_spectrum import get_light_universe, LightSpectrum
from Core.S1_Body.L5_Mental.Digestion.universal_digestor import get_universal_digestor, RawKnowledgeChunk, ChunkType
from Core.S1_Body.L5_Mental.Memory.kg_manager import get_kg_manager

logger = logging.getLogger("CumulativeDigestor")

class CumulativeDigestor:
    """
    Handles the systematic accumulation of knowledge from the codebase and documentation.
    """
    def __init__(self, root_path: str = "c:/Elysia"):
        self.root_path = Path(root_path)
        self.universe = get_light_universe()
        self.digestor = get_universal_digestor()
        self.kg = get_kg_manager()
        
    def digest_docs(self, docs_dir: str = "docs"):
        """
        Fast batch digestion of documentation.
        """
        doc_path = self.root_path / docs_dir
        if not doc_path.exists():
            logger.warning(f"⚠️ Docs directory not found: {doc_path}")
            return

        logger.info(f"🌿 [CUMULATIVE_DIGESTOR] Projecting documentation at {doc_path}...")
        
        entries = []
        for root, dirs, files in os.walk(doc_path):
            dirs[:] = [d for d in dirs if d not in ['__pycache__', 'node_modules']]
            for file in files:
                if file.endswith(".md") or file.endswith(".txt"):
                    full_path = Path(root) / file
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        rel_path = full_path.relative_to(self.root_path)
                        # Scale 1 (Space) for documentation context
                        entries.append((content, f"doc:{rel_path}", 1))
                    except Exception as e:
                        logger.error(f"   Failed to read {file}: {e}")

        logger.info(f"✨ [CUMULATIVE_DIGESTOR] {len(entries)} documents collected for projection.")

        # Instantaneous field projection
        # Stratum 2: Line / Intellectual Library
        self.universe.batch_absorb(entries, stratum=2)
        self.universe.save_state()

        # [NEW: Autonomous Causal Digestion]
        # We also digest the content into the Knowledge Graph for structural understanding
        logger.info(f"🧬 [CUMULATIVE_DIGESTOR] Distilling causal structures into the KG...")
        total_nodes = 0
        for content, tag, scale in entries:
            chunk = RawKnowledgeChunk(
                chunk_id=f"digest_{tag.replace(':', '_')}",
                chunk_type=ChunkType.TEXT,
                content=content,
                source=tag
            )
            nodes = self.digestor.digest(chunk)
            total_nodes += len(nodes)
            for node in nodes:
                self.kg.add_node(node.concept.lower(), properties={"source": tag})
                for rel in node.relations:
                    self.kg.add_edge(node.concept.lower(), rel.lower(), "resonates_with")
        
        self.kg.save()
        logger.info(f"✨ [CUMULATIVE_DIGESTOR] {len(entries)} documents projected. {total_nodes} nodes distilled.")

    def digest_vocabulary(self, terms: List[Dict[str, str]]):
        """
        Explicitly digests a list of vocabulary terms.
        Format: [{'term': 'Resonance', 'description': '...'}]
        """
        for item in terms:
            term = item.get("term", "")
            desc = item.get("description", "")
            if term:
                self.universe.absorb_with_terrain(
                    f"{term}: {desc}",
                    tag=f"vocab:{term}",
                    scale=2, # Line basis for definitions/relations
                    stratum=2 # Stratum 2: Intellectual Library
                )
        self.universe.save_state()
        logger.info(f"✨ [CUMULATIVE_DIGESTOR] {len(terms)} vocabulary terms sedimented.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    digestor = CumulativeDigestor()
    digestor.digest_docs()
