from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict
import numpy as np
from Project_Sophia.core.tensor_wave import Tensor3D, FrequencyWave

@dataclass
class Thought:
    """
    Represents a single unit of thought or insight within the system.
    It standardizes the structure of facts deduced from various sources,
    enabling richer and more nuanced decision-making.

    Now Supports Fractal Recursion:
    - A thought can contain other thoughts (`sub_thoughts`).
    - A thought calculates its own 'Gravitational Mass' based on this depth.
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
    # The 'spin' (within tensor) gives it drilling power (Rifling).
    tensor: Tensor3D = field(default_factory=Tensor3D)
    wave: FrequencyWave = field(default_factory=lambda: FrequencyWave(0.0, 0.0, 0.0, 0.0))

    # --- Recursive Structure ---
    sub_thoughts: List['Thought'] = field(default_factory=list) # The internal micro-cosmos of this thought
    orbiting_concepts: List[str] = field(default_factory=list) # Concepts attracted to this thought's gravity

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

        # --- Calculate Initial Rifling (Spin) ---
        # If this thought comes from deep intuition (high frequency wave) or strong emotion,
        # we verify it has appropriate spin.
        if self.wave.frequency > 400 and self.tensor.spin_magnitude() == 0:
             # High frequency thoughts naturally spin.
             # We call distribute_frequency to get a template tensor with spin,
             # then apply that spin to our current tensor.
             template = Tensor3D.distribute_frequency(self.wave.frequency)
             self.tensor.spin = template.spin

        # --- Fractal Mass Calculation ---
        # Update the tensor's mass offset based on recursive depth
        self.tensor.mass_offset += self.calculate_gravitational_mass() * 0.1

    @property
    def rifling_power(self) -> float:
        """
        Returns the 'drilling capability' of this thought.
        Used by the CognitionPipeline to determine if this thought can 'tunnel'
        through confusion or skip logical steps (Hyperdrive).
        """
        return self.tensor.calculate_rifling()

    def calculate_gravitational_mass(self) -> float:
        """
        Calculates the 'mass' of this thought.
        Mass = Base Energy + (Count of Sub-Thoughts * Recursive Factor).
        A deep, complex thought has higher gravity than a shallow one.
        """
        base_mass = self.energy + (self.confidence * 2.0)
        recursive_mass = sum(t.calculate_gravitational_mass() for t in self.sub_thoughts) * 0.5
        evidence_mass = len(self.evidence) * 0.2
        return base_mass + recursive_mass + evidence_mass

    def __str__(self):
        rifling = self.rifling_power
        mass = self.calculate_gravitational_mass()
        return (f"Thought(content='{self.content}', source='{self.source}', "
                f"confidence={self.confidence:.2f}, energy={self.energy:.2f}, "
                f"rifling={rifling:.2f}, mass={mass:.2f}, wave={self.wave}, subs={len(self.sub_thoughts)})")
