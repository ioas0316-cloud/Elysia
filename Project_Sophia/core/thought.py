from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class Thought:
    """
    Represents a single unit of thought or insight within the system.
    It standardizes the structure of facts deduced from various sources,
    enabling richer and more nuanced decision-making.
    """
    content: str
    source: str  # e.g., 'knowledge_graph', 'living_reason_system', 'memory'
    confidence: float = 0.9  # The certainty of the thought's validity (0.0 to 1.0)
    energy: float = 0.0      # The activation energy from the cell simulation
    evidence: List[Any] = field(default_factory=list) # Supporting nodes/edges/cells

    # --- Soul Layer Attributes (Frequency/Resonance) ---
    frequency: float = 0.0   # The fundamental tone (Hz) of this thought. (New in Fractal Soul update)
    resonance_amp: float = 0.0 # The intensity of the resonance (Amplitude).

    def __str__(self):
        return f"Thought(content='{self.content}', source='{self.source}', confidence={self.confidence:.2f}, energy={self.energy:.2f}, freq={self.frequency:.1f}Hz)"
