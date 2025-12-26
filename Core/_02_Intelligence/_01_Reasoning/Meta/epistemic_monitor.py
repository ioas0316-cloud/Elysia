"""
Epistemic Monitor: The Quality of Knowing
=========================================

"To know that you do not know is the best." - Lao Tzu

Purpose:
Tracks the source (Provenance) of all knowledge in the HyperGraph.
Calculates 'Epistemic Uncertainty' to prevent hallucination.

Sources:
- AXIOMATIC (Code/Logic) -> High Certainty
- SENSORY (Input) -> High Certainty
- INFERRED (Reasoning) -> Medium Certainty
- SIMULATED (Hallucination) -> LOW Certainty (Ignorance)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any
from Core._01_Foundation._01_Infrastructure.elysia_core import Cell

class EpistemicSource(Enum):
    AXIOMATIC = "axiomatic"   # Hardcoded truths, math
    SENSORY = "sensory"       # Direct input (User, Camera, Mic)
    INFERRED = "inferred"     # Deducing B from A
    SIMULATED = "simulated"   # Placeholder / Guess
    VOID = "void"             # Complete Ignorance

@dataclass
class KnowledgeQuality:
    source: EpistemicSource
    uncertainty: float  # 0.0 (Fact) to 1.0 (Pure Guess)
    timestamp: float

@Cell("EpistemicMonitor")
class EpistemicMonitor:
    """
    The Conscience of the System.
    Rejects 'Simulated' knowledge if better sources are available.
    """
    
    def evaluate_knowledge(self, node: Any) -> KnowledgeQuality:
        """Analyze a node to determine its Truthiness."""
        # Simple heuristic for now
        if not hasattr(node, "definition") or not node.definition:
            return KnowledgeQuality(EpistemicSource.VOID, 1.0, 0)
            
        text = node.definition.lower()
        
        if "[simulated]" in text or "simulated insight" in text:
            return KnowledgeQuality(EpistemicSource.SIMULATED, 0.9, 0)
            
        if "user said" in text or "input:" in text:
            return KnowledgeQuality(EpistemicSource.SENSORY, 0.1, 0)
            
        return KnowledgeQuality(EpistemicSource.INFERRED, 0.5, 0)

    def is_hallucination(self, node: Any) -> bool:
        """Returns True if the knowledge is fake/simulated."""
        quality = self.evaluate_knowledge(node)
        return quality.source == EpistemicSource.SIMULATED
