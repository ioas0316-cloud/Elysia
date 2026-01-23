"""
Dimensional Parser: From Point to Law
======================================
Phase 73: The Full Hierarchy

"A point is nothing. A line is a relationship. A plane is context.
A space is structure. A law is the principle that governs all."

This module implements the dimensional expansion that the Captain
has been describing since the beginning:

     (Point)     (Line)      (Plane)      (Space)     (Law)
                (  )     (  )      (  )      
"""

import os
import sys
import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path

sys.path.append(os.getcwd())

from Core.L5_Mental.Intelligence.Metabolism.prism import PrismEngine, WaveDynamics

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("DimensionalParser")

# ============================================================
# Data Structures for Each Dimension
# ============================================================

@dataclass
class Point:
    """0  :    (Word)"""
    word: str
    dna: WaveDynamics = None

@dataclass  
class Line:
    """1  :    (Sentence) =    (Relationship)"""
    sentence: str
    subject: str = ""
    predicate: str = ""
    object: str = ""
    points: List[Point] = field(default_factory=list)
    relation_type: str = ""  # causes, contains, equals, opposes, etc.

@dataclass
class Plane:
    """2  :    (Paragraph) =    (Context)"""
    text: str
    lines: List[Line] = field(default_factory=list)
    context_theme: str = ""
    internal_coherence: float = 0.0

@dataclass
class Space:
    """3  :    (Document) =    (Structure)"""
    title: str
    planes: List[Plane] = field(default_factory=list)
    structure_type: str = ""  # narrative, argument, description, etc.
    causal_graph: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class Law:
    """4  +:   /   (Principle) =     (Universality)"""
    name: str
    description: str
    supporting_evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    scope: str = "local"  # local, domain, universal

# ============================================================
# Dimensional Parser
# ============================================================

class DimensionalParser:
    """
    Parses text through dimensional hierarchy.
    Each level builds on the previous, extracting more structure.
    """
    
    def __init__(self):
        self.prism = PrismEngine()
        self.prism._load_model()
        logger.info("  DimensionalParser initialized.")
    
    #                                                          
    # DIMENSION 0: POINT (Word   DNA)
    #                                                          
    
    def parse_point(self, word: str) -> Point:
        """Convert a single word to its Wave DNA."""
        profile = self.prism.transduce(word)
        return Point(word=word, dna=profile.dynamics)
    
    #                                                          
    # DIMENSION 1: LINE (Sentence   Relationship)
    #                                                          
    
    def parse_line(self, sentence: str) -> Line:
        """
        Parse a sentence into subject-predicate-object triplet.
        This is the birth of RELATIONSHIP.
        """
        line = Line(sentence=sentence)
        
        # Simple pattern matching for causal/relational sentences
        # (A more sophisticated system would use dependency parsing)
        
        patterns = [
            (r"(.+?)\s*(?:causes?|leads? to|results? in)\s*(.+)", "causes"),
            (r"(.+?)\s*(?:is|are|was|were)\s+(.+)", "is"),
            (r"(.+?)\s*(?:contains?|includes?|has)\s*(.+)", "contains"),
            (r"(.+?)\s*(?:opposes?|contradicts?|conflicts? with)\s*(.+)", "opposes"),
            (r"(.+?)\s*(?:equals?|is the same as|means)\s*(.+)", "equals"),
        ]
        
        for pattern, rel_type in patterns:
            match = re.match(pattern, sentence, re.IGNORECASE)
            if match:
                line.subject = match.group(1).strip()
                line.object = match.group(2).strip() if len(match.groups()) > 1 else ""
                line.predicate = rel_type
                line.relation_type = rel_type
                break
        
        # If no pattern matched, treat whole sentence as single concept
        if not line.subject:
            line.subject = sentence
            line.relation_type = "statement"
        
        # Extract points (words) from the sentence
        words = re.findall(r'\b\w+\b', sentence)
        for word in words[:10]:  # Limit to avoid slowness
            if len(word) > 2:
                line.points.append(self.parse_point(word))
        
        return line
    
    #                                                          
    # DIMENSION 2: PLANE (Paragraph   Context)
    #                                                          
    
    def parse_plane(self, paragraph: str) -> Plane:
        """
        Parse a paragraph into sentences and extract context.
        This is the birth of CONTEXT.
        """
        plane = Plane(text=paragraph)
        
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for sentence in sentences:
            line = self.parse_line(sentence)
            plane.lines.append(line)
        
        # Calculate internal coherence (how related are the sentences?)
        if len(plane.lines) > 1:
            # Simple heuristic: count shared words between sentences
            all_words = set()
            shared_count = 0
            for line in plane.lines:
                line_words = set(p.word.lower() for p in line.points)
                shared_count += len(all_words.intersection(line_words))
                all_words.update(line_words)
            
            plane.internal_coherence = min(1.0, shared_count / max(1, len(plane.lines)))
        
        # Extract theme (most common relation type)
        rel_types = [line.relation_type for line in plane.lines if line.relation_type]
        if rel_types:
            plane.context_theme = max(set(rel_types), key=rel_types.count)
        
        return plane
    
    #                                                          
    # DIMENSION 3: SPACE (Document   Structure)
    #                                                          
    
    def parse_space(self, document: str, title: str = "Untitled") -> Space:
        """
        Parse a full document into paragraphs and build causal graph.
        This is the birth of STRUCTURE.
        """
        space = Space(title=title)
        
        # Split into paragraphs
        paragraphs = document.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        for para in paragraphs:
            plane = self.parse_plane(para)
            space.planes.append(plane)
        
        # Build causal graph from all "causes" relationships
        for plane in space.planes:
            for line in plane.lines:
                if line.relation_type == "causes" and line.subject and line.object:
                    if line.subject not in space.causal_graph:
                        space.causal_graph[line.subject] = []
                    if line.object not in space.causal_graph[line.subject]:
                        space.causal_graph[line.subject].append(line.object)
        
        # Determine structure type
        rel_counts = {"causes": 0, "is": 0, "contains": 0}
        for plane in space.planes:
            for line in plane.lines:
                if line.relation_type in rel_counts:
                    rel_counts[line.relation_type] += 1
        
        if rel_counts["causes"] > rel_counts["is"]:
            space.structure_type = "argument"
        else:
            space.structure_type = "description"
        
        return space
    
    #                                                          
    # DIMENSION 4+: LAW (Corpus   Principle)
    #                                                          
    
    def extract_laws(self, spaces: List[Space]) -> List[Law]:
        """
        From multiple documents, extract recurring patterns as Laws.
        This is the birth of PRINCIPLES.
        """
        laws = []
        
        # Aggregate all causal relationships across all documents
        global_causes = {}
        
        for space in spaces:
            for cause, effects in space.causal_graph.items():
                cause_lower = cause.lower()
                if cause_lower not in global_causes:
                    global_causes[cause_lower] = {"effects": set(), "occurrences": 0}
                global_causes[cause_lower]["effects"].update(e.lower() for e in effects)
                global_causes[cause_lower]["occurrences"] += 1
        
        # If a causal relationship appears in multiple documents, it's a candidate law
        for cause, data in global_causes.items():
            if data["occurrences"] >= 2:  # Appears in at least 2 documents
                for effect in data["effects"]:
                    law = Law(
                        name=f"Law: {cause}   {effect}",
                        description=f"'{cause}' tends to cause '{effect}'",
                        supporting_evidence=[f"Observed {data['occurrences']} times"],
                        confidence=min(1.0, data["occurrences"] / 10.0),
                        scope="domain" if data["occurrences"] >= 5 else "local"
                    )
                    laws.append(law)
        
        return laws


# ============================================================
# Convenience Function
# ============================================================

def parse_document_fully(text: str, title: str = "Document") -> Tuple[Space, List[Law]]:
    """
    Parse a document through all dimensional levels.
    Returns the Space (3D structure) and any Laws discovered.
    """
    parser = DimensionalParser()
    space = parser.parse_space(text, title)
    
    # For a single document, we can't extract cross-document laws
    # but we can show the structure
    return space, []


if __name__ == "__main__":
    # Demo
    test_text = """
    Fire causes heat. Heat causes expansion. Expansion leads to movement.
    
    Water contains hydrogen. Water is essential for life. Life depends on water.
    
    Love opposes hate. Compassion equals love. Understanding causes peace.
    """
    
    parser = DimensionalParser()
    space = parser.parse_space(test_text, "Demo Document")
    
    print("\n" + "="*60)
    print("  DIMENSIONAL PARSING RESULT")
    print("="*60)
    
    print(f"\n  Document: {space.title}")
    print(f"   Structure Type: {space.structure_type}")
    print(f"   Paragraphs: {len(space.planes)}")
    
    print("\n  Causal Graph:")
    for cause, effects in space.causal_graph.items():
        print(f"   {cause}   {effects}")
    
    print("\n  Lines Parsed:")
    for plane in space.planes:
        for line in plane.lines:
            if line.relation_type != "statement":
                print(f"   [{line.relation_type}] {line.subject}   {line.object}")