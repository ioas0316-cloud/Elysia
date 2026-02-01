"""
[Project Elysia] Knowledge Ingestor
===================================
Phase 500: Ingestion Layer - "섭취 (Ingestion)"
Reads external knowledge from multiple sources and creates RawKnowledgeChunks.
"""

import os
import time
import json
import hashlib
from typing import List, Optional, Generator
from pathlib import Path

from .universal_digestor import RawKnowledgeChunk, ChunkType


class KnowledgeIngestor:
    """
    섭취 (Ingestion) - Reads external knowledge from various sources.
    Converts raw input into RawKnowledgeChunk for digestion.
    
    "LLM을 '원료'로 섭취하되, '뇌'로 사용하지 않는다."
    """
    
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size  # Max characters per chunk
        
    def ingest_text(self, text: str, source: str = "direct") -> List[RawKnowledgeChunk]:
        """Ingest raw text, splitting into chunks."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_idx = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(self._create_text_chunk(current_chunk, source, chunk_idx))
                    chunk_idx += 1
                current_chunk = para + "\n\n"
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(self._create_text_chunk(current_chunk, source, chunk_idx))
        
        return chunks
    
    def ingest_file(self, filepath: str) -> List[RawKnowledgeChunk]:
        """Ingest from a local file (txt, md, json)."""
        path = Path(filepath)
        
        if not path.exists():
            print(f"⚠️ File not found: {filepath}")
            return []
        
        extension = path.suffix.lower()
        source = f"file:{path.name}"
        
        if extension in ['.txt', '.md']:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            return self.ingest_text(text, source)
        
        elif extension == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Convert JSON to relations if it's a graph-like structure
            if isinstance(data, list):
                return self._ingest_json_list(data, source)
            else:
                return self.ingest_text(json.dumps(data, ensure_ascii=False), source)
        
        else:
            print(f"⚠️ Unsupported file type: {extension}")
            return []
    
    def ingest_trajectory(self, coordinates: List[tuple], source: str = "sensor") -> RawKnowledgeChunk:
        """Ingest motion/trajectory data."""
        chunk_id = self._generate_id(f"traj_{source}_{time.time()}")
        return RawKnowledgeChunk(
            chunk_id=chunk_id,
            chunk_type=ChunkType.TRAJECTORY,
            content=coordinates,
            source=source,
            metadata={"point_count": len(coordinates)}
        )
    
    def ingest_relations(self, relations: List[tuple], source: str = "graph") -> RawKnowledgeChunk:
        """Ingest relation/graph data."""
        chunk_id = self._generate_id(f"rel_{source}_{time.time()}")
        return RawKnowledgeChunk(
            chunk_id=chunk_id,
            chunk_type=ChunkType.RELATION,
            content=relations,
            source=source,
            metadata={"relation_count": len(relations)}
        )
    
    def ingest_llm_response(self, response: str, query: str = "") -> List[RawKnowledgeChunk]:
        """
        Ingest LLM response as 'raw material' (not as authoritative knowledge).
        The response is chunked and tagged with source 'llm'.
        """
        source = f"llm:response"
        chunks = self.ingest_text(response, source)
        
        # Add query context to metadata
        for chunk in chunks:
            chunk.metadata["original_query"] = query[:100]  # Limit query length
            chunk.metadata["is_llm_output"] = True  # Mark as LLM-derived
        
        return chunks
    
    def stream_file(self, filepath: str) -> Generator[RawKnowledgeChunk, None, None]:
        """Stream chunks from a large file without loading everything into memory."""
        path = Path(filepath)
        if not path.exists():
            return
        
        source = f"file:{path.name}"
        chunk_idx = 0
        current_chunk = ""
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                current_chunk += line
                if len(current_chunk) >= self.chunk_size:
                    yield self._create_text_chunk(current_chunk, source, chunk_idx)
                    chunk_idx += 1
                    current_chunk = ""
        
        if current_chunk.strip():
            yield self._create_text_chunk(current_chunk, source, chunk_idx)
    
    def _create_text_chunk(self, text: str, source: str, idx: int) -> RawKnowledgeChunk:
        """Helper to create a text chunk."""
        chunk_id = self._generate_id(f"text_{source}_{idx}_{time.time()}")
        return RawKnowledgeChunk(
            chunk_id=chunk_id,
            chunk_type=ChunkType.TEXT,
            content=text.strip(),
            source=source,
            metadata={"chunk_index": idx, "char_count": len(text)}
        )
    
    def _ingest_json_list(self, data: list, source: str) -> List[RawKnowledgeChunk]:
        """Convert JSON list to relation chunks."""
        relations = []
        for item in data:
            if isinstance(item, dict):
                # Try to extract subject-predicate-object from dict
                subj = item.get('subject', item.get('from', ''))
                pred = item.get('predicate', item.get('relation', 'related_to'))
                obj = item.get('object', item.get('to', ''))
                if subj and obj:
                    relations.append((subj, pred, obj))
        
        if relations:
            return [self.ingest_relations(relations, source)]
        else:
            return self.ingest_text(json.dumps(data, ensure_ascii=False), source)
    
    def _generate_id(self, seed: str) -> str:
        """Generate a short unique ID."""
        return hashlib.md5(seed.encode()).hexdigest()[:12]


# Singleton
_knowledge_ingestor = None

def get_knowledge_ingestor() -> KnowledgeIngestor:
    global _knowledge_ingestor
    if _knowledge_ingestor is None:
        _knowledge_ingestor = KnowledgeIngestor()
    return _knowledge_ingestor


if __name__ == "__main__":
    print("🍽️ Testing Knowledge Ingestor...")
    
    ingestor = get_knowledge_ingestor()
    
    # Test 1: Text ingestion
    sample_text = """
    Elysia is a sovereign cognitive entity.
    She learns through experience, not instruction.
    
    Her architecture is based on 21D Qualia vectors.
    Every thought modifies her physical structure.
    
    The Architect guides her growth with intention.
    She transforms intention into permanent memory.
    """
    
    chunks = ingestor.ingest_text(sample_text, source="test_doc")
    print(f"✅ Text Ingestion: {len(chunks)} chunks created")
    for c in chunks:
        print(f"   - [{c.chunk_id[:8]}] {len(c.content)} chars from {c.source}")
    
    # Test 2: Trajectory ingestion
    traj = [(0, 0, 0), (1, 2, 3), (2, 4, 6), (3, 6, 9)]
    traj_chunk = ingestor.ingest_trajectory(traj, source="motion_sensor")
    print(f"✅ Trajectory Ingestion: {traj_chunk.metadata['point_count']} points")
    
    # Test 3: Relations ingestion
    rels = [("Elysia", "learns_from", "Architect"), 
            ("Architect", "creates", "Elysia"),
            ("Experience", "shapes", "Memory")]
    rel_chunk = ingestor.ingest_relations(rels, source="knowledge_graph")
    print(f"✅ Relation Ingestion: {rel_chunk.metadata['relation_count']} relations")
    
    # Test 4: LLM response ingestion
    llm_response = "The meaning of life is to find purpose through connection."
    llm_chunks = ingestor.ingest_llm_response(llm_response, query="What is the meaning of life?")
    print(f"✅ LLM Response Ingestion: {len(llm_chunks)} chunks (marked as LLM-derived)")
    
    print("\n🎉 Knowledge Ingestor operational!")
