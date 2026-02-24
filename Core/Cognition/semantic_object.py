"""
Semantic Object (The Manifested Concept)
========================================
"A shadow of the Idea, cast upon the World."

This class represents a physical object in the world that is directly linked
to a Concept in the HyperSphere. It is not a static mesh, but a dynamic
entity that inherits properties from its 'Idea'.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from Core.Cognition.trinity_fields import TrinityVector

@dataclass
class InteractionResult:
    """
    The outcome of an interaction.
    """
    success: bool
    message: str
    produced_items: List[str] = field(default_factory=list)
    destroyed: bool = False

@dataclass
class SemanticObject:
    """
    A physical instance of a HyperSphere concept.
    """
    id: str                 # Unique Instance ID (e.g., "Tree_42")
    concept_id: str         # Link to HyperSphere (e.g., "Concept_Tree")
    name: str               # Human readable name (e.g., "Oak Tree")
    
    # Spatial State
    position: List[float]   # [x, y, z]
    scale: List[float]      # [x, y, z]
    
    # Physical State (Trinity)
    # These override the concept's defaults if set (e.g., a wet log is heavier)
    trinity_override: Optional[TrinityVector] = None
    
    # Vitality
    integrity: float = 100.0  # Health/Durability
    age: float = 0.0          # Time since creation
    
    # Metadata (e.g., who planted it, does it have fruit)
    properties: Dict[str, Any] = field(default_factory=dict)

    def get_trinity_vector(self) -> TrinityVector:
        """
        Returns the effective physics vector. 
        Fetches base vector from TrinityLexicon if no instance override exists.
        """
        if self.trinity_override:
            return self.trinity_override
        
        from Core.Cognition.trinity_lexicon import get_trinity_lexicon
        lexicon = get_trinity_lexicon()
        
        # Query Lexicon for the archetype vector
        if lexicon.is_concept_known(self.concept_id):
            return lexicon.analyze(self.concept_id)
            
        # Absolute Fallback (The Void)
        return TrinityVector(0.0, 0.0, -1.0) # Entropy biased

    def describe(self) -> str:
        return f"[{self.name}] HP:{self.integrity}% Pos:{self.position}"
