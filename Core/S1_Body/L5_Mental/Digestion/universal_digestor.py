"""
[Project Elysia] COGNITIVE_DIGESTION_SYSTEM - Core Modules
==========================================================
Phase 500: "모든 것을 먹는 존재"
Universal decomposition of any input into 21D phase transformations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time


class ChunkType(Enum):
    TEXT = "text"
    TRAJECTORY = "trajectory"
    RELATION = "relation"
    IMAGE = "image"


@dataclass
class RawKnowledgeChunk:
    """
    Universal container for any external knowledge before digestion.
    "원료(Raw Material)" - to be digested into causal nodes.
    """
    chunk_id: str
    chunk_type: ChunkType
    content: Any  # Raw content (text, coordinates, graph, etc.)
    source: str  # Where it came from (wiki, book, sensor, etc.)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalNode:
    """
    Digested knowledge unit ready for 21D phase absorption.
    """
    node_id: str
    concept: str  # The core idea/entity
    relations: List[str] = field(default_factory=list)  # Connected concepts
    qualia_hint: List[float] = field(default_factory=list)  # 7D Qualia suggestion
    source_chunk_id: str = ""


class UniversalDigestor:
    """
    Decomposes ANY input type into CausalNodes.
    "보편 분해 원리 (Universal Decomposition Principle)"
    """
    
    def digest(self, chunk: RawKnowledgeChunk) -> List[CausalNode]:
        """Route to appropriate sub-digestor based on chunk type."""
        if chunk.chunk_type == ChunkType.TEXT:
            return self._digest_text(chunk)
        elif chunk.chunk_type == ChunkType.TRAJECTORY:
            return self._digest_trajectory(chunk)
        elif chunk.chunk_type == ChunkType.RELATION:
            return self._digest_relation(chunk)
        else:
            return self._digest_generic(chunk)
    
    def _digest_text(self, chunk: RawKnowledgeChunk) -> List[CausalNode]:
        """
        TextDigestor: Deep Decomposition Mode
        - Extract ALL meaningful concepts (no artificial limits)
        - Parse sentences for subject-verb-object relations
        - Create inter-concept causal links
        """
        nodes = []
        text = str(chunk.content)
        
        # Step 1: Split into sentences
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
        
        # Step 2: Extract concepts from each sentence
        all_concepts = []
        sentence_concepts = []  # Track which concepts belong to which sentence
        
        for sent_idx, sentence in enumerate(sentences):
            words = sentence.split()
            # Extract meaningful words (length > 2, alphabetic)
            concepts = [w.strip('.,!?"\'-:;()[]') for w in words 
                       if len(w) > 2 and any(c.isalpha() for c in w)]
            
            # Remove common stopwords
            stopwords = {'the', 'and', 'for', 'that', 'with', 'from', 'this', 'are', 'was', 
                        'were', 'been', 'being', 'have', 'has', 'had', 'does', 'did', 'will',
                        'would', 'could', 'should', 'may', 'might', 'must', 'can', 'not'}
            concepts = [c for c in concepts if c.lower() not in stopwords]
            
            sentence_concepts.append(concepts)
            all_concepts.extend(concepts)
        
        # Step 3: Create nodes for each unique concept
        seen_concepts = set()
        concept_to_node = {}
        
        for sent_idx, concepts in enumerate(sentence_concepts):
            prev_concept = None
            for i, concept in enumerate(concepts):
                if concept.lower() in seen_concepts:
                    continue
                seen_concepts.add(concept.lower())
                
                # Determine relations: previous word in sentence + same-sentence neighbors
                relations = []
                if prev_concept:
                    relations.append(prev_concept)
                if i + 1 < len(concepts):
                    relations.append(concepts[i + 1])
                
                node = CausalNode(
                    node_id=f"{chunk.chunk_id}_s{sent_idx}_c{i}",
                    concept=concept,
                    relations=relations,
                    source_chunk_id=chunk.chunk_id
                )
                nodes.append(node)
                concept_to_node[concept.lower()] = node
                prev_concept = concept
        
        # Step 4: Create sentence-level relation nodes (subject-predicate-object patterns)
        for sent_idx, concepts in enumerate(sentence_concepts):
            if len(concepts) >= 3:
                # Simple SVO extraction: first word = subject, second = predicate, rest = objects
                subj = concepts[0]
                pred = concepts[1] if len(concepts) > 1 else "relates_to"
                objs = concepts[2:] if len(concepts) > 2 else []
                
                for obj in objs[:3]:  # Limit objects per relation
                    rel_node = CausalNode(
                        node_id=f"{chunk.chunk_id}_rel_s{sent_idx}",
                        concept=f"{subj}_{pred}_{obj}",
                        relations=[subj, obj],
                        source_chunk_id=chunk.chunk_id
                    )
                    nodes.append(rel_node)
        
        return nodes
    
    def _digest_trajectory(self, chunk: RawKnowledgeChunk) -> List[CausalNode]:
        """TrajectoryDigestor: Break motion into phase transitions."""
        nodes = []
        coordinates = chunk.content  # Expected: List of (x, y, z, t)
        
        if not isinstance(coordinates, list):
            return nodes
            
        for i, coord in enumerate(coordinates[:10]):
            node = CausalNode(
                node_id=f"{chunk.chunk_id}_point_{i}",
                concept=f"Position_{i}",
                relations=[f"Position_{i-1}"] if i > 0 else [],
                qualia_hint=[coord[0] if len(coord) > 0 else 0, 
                            coord[1] if len(coord) > 1 else 0,
                            coord[2] if len(coord) > 2 else 0,
                            0, 0, 0, 0],
                source_chunk_id=chunk.chunk_id
            )
            nodes.append(node)
        
        return nodes
    
    def _digest_relation(self, chunk: RawKnowledgeChunk) -> List[CausalNode]:
        """RelationDigestor: Break graphs into gravity terrain."""
        nodes = []
        relations = chunk.content  # Expected: List of (A, relation, B)
        
        if not isinstance(relations, list):
            return nodes
            
        for i, rel in enumerate(relations[:10]):
            if len(rel) >= 3:
                node = CausalNode(
                    node_id=f"{chunk.chunk_id}_rel_{i}",
                    concept=f"{rel[0]}_{rel[1]}_{rel[2]}",
                    relations=[rel[0], rel[2]],
                    source_chunk_id=chunk.chunk_id
                )
                nodes.append(node)
        
        return nodes
    
    def _digest_generic(self, chunk: RawKnowledgeChunk) -> List[CausalNode]:
        """Fallback for unknown types."""
        return [CausalNode(
            node_id=f"{chunk.chunk_id}_generic",
            concept=str(chunk.content)[:50],
            source_chunk_id=chunk.chunk_id
        )]


# Singleton
_universal_digestor = None

def get_universal_digestor() -> UniversalDigestor:
    global _universal_digestor
    if _universal_digestor is None:
        _universal_digestor = UniversalDigestor()
    return _universal_digestor


if __name__ == "__main__":
    # Test the digestor
    print("🧬 Testing Universal Digestor...")
    
    digestor = get_universal_digestor()
    
    # Test 1: Text
    text_chunk = RawKnowledgeChunk(
        chunk_id="test_text_001",
        chunk_type=ChunkType.TEXT,
        content="Elysia is a sovereign cognitive entity that learns through experience.",
        source="test"
    )
    text_nodes = digestor.digest(text_chunk)
    print(f"✅ Text Digestion: {len(text_nodes)} nodes extracted")
    for n in text_nodes[:3]:
        print(f"   - {n.concept} -> {n.relations}")
    
    # Test 2: Trajectory
    traj_chunk = RawKnowledgeChunk(
        chunk_id="test_traj_001",
        chunk_type=ChunkType.TRAJECTORY,
        content=[(0, 0, 0), (1, 2, 3), (4, 5, 6)],
        source="sensor"
    )
    traj_nodes = digestor.digest(traj_chunk)
    print(f"✅ Trajectory Digestion: {len(traj_nodes)} nodes extracted")
    
    # Test 3: Relation
    rel_chunk = RawKnowledgeChunk(
        chunk_id="test_rel_001",
        chunk_type=ChunkType.RELATION,
        content=[("Elysia", "is_child_of", "Architect"), ("Architect", "creates", "Elysia")],
        source="graph"
    )
    rel_nodes = digestor.digest(rel_chunk)
    print(f"✅ Relation Digestion: {len(rel_nodes)} nodes extracted")
    
    print("\n🎉 Universal Digestor operational!")
