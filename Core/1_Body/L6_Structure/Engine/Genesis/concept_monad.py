"""
Core/Engine/Genesis/concept_monad.py
=================================
The Substance of ANY Thing.

A Monad can be an Electron, a Dollar, or a Feeling.
"""
import uuid
from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class ConceptMonad:
    """
    The Universal Particle.
    """
    name: str
    domain: str # 'Physics', 'Economy', 'Mind'
    val: float  # Mass, Price, Intensity
    
    # Dynamic Properties (The DNA)
    props: Dict[str, Any] = field(default_factory=dict)
    
    # Unique ID
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __repr__(self):
        return f"<{self.domain}:{self.name} val={self.val:.2f}>"
