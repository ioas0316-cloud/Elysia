from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .tensor_wave import Tensor3D

@dataclass
class Thought:
    """
    Represents a discrete unit of thought (a 'Concept' or 'Fact')
    wrapped in a quantum tensor state.
    """
    content: str
    source: str  # e.g., 'bone' (static), 'flesh' (simulated), 'spirit' (inspired)
    confidence: float
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    energy: float = 0.0
    tensor: Optional[Tensor3D] = None
    vcd_score: float = 0.0 # Value-Centered Decision score

    def __post_init__(self):
        if self.tensor is None:
            self.tensor = Tensor3D() # Default neutral state
