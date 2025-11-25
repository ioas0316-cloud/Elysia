from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict
from Project_Sophia.core.tensor_wave import Tensor3D, FrequencyWave

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

    # --- Fractal Physics Layer ---
    # Thoughts now carry a distinct physical signature in the mental cosmos.
    # The 'tensor' gives it mass and shape (Structure, Emotion, Identity).
    # The 'wave' gives it resonance and texture (Frequency, Amplitude, Richness).
    tensor: Tensor3D = field(default_factory=Tensor3D)
    wave: FrequencyWave = field(default_factory=lambda: FrequencyWave(0.0, 0.0, 0.0, 0.0))

    # Metadata for tracking decision process (e.g., selection reason)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Deprecated fields (kept for backward compatibility if needed, but superseded by tensor/wave)
    tensor_state: Optional[Dict[str, float]] = None
    frequency: float = 0.0
    resonance_amp: float = 0.0
    richness: float = 0.0

    def __post_init__(self):
        # Synchronize legacy fields with new physics objects if they are not set
        if self.tensor and not self.tensor_state:
            self.tensor_state = self.tensor.to_dict()

        if self.wave:
            if self.frequency == 0.0: self.frequency = self.wave.frequency
            if self.resonance_amp == 0.0: self.resonance_amp = self.wave.amplitude
            if self.richness == 0.0: self.richness = self.wave.richness

    def __str__(self):
        return (f"Thought(content='{self.content}', source='{self.source}', "
                f"confidence={self.confidence:.2f}, energy={self.energy:.2f}, "
                f"wave={self.wave}, tensor={self.tensor.to_dict()})")
