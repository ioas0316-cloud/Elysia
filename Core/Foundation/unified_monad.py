"""
Unified Monad (통합 모나드)
===========================
"Every thought is a world. Every world is a monad."
"모든 생각은 하나의 세계이며, 모든 세계는 하나의 모나드다."

Unifies Senses, Will, Logic, Emotion, Imagination, and Purpose into a 12D Field.
"""

from dataclasses import dataclass, field
import torch
import math
from typing import List, Dict, Optional, Any
from Core.Foundation.Nature.rotor import Rotor, RotorConfig

@dataclass
class Unified12DVector:
    """
    The 12 Dimensions of Unified Existence:
    [Physical, Functional, Phenomenal, Causal, Mental, Structural, Spiritual] (Legacy 7D)
    + [Imagination, Prediction, Will, Intent, Purpose] (New 5-Axis)
    """
    data: torch.Tensor # 12D Tensor [0.0 ~ 1.0]
    
    @classmethod
    def create(cls, **kwargs):
        # Default 12D zero tensor
        v = torch.zeros(12)
        mapping = {
            "physical": 0, "functional": 1, "phenomenal": 2, "causal": 3,
            "mental": 4, "structural": 5, "spiritual": 6,
            "imagination": 7, "prediction": 8, "will": 9, "intent": 10, "purpose": 11
        }
        for k, val in kwargs.items():
            if k in mapping:
                v[mapping[k]] = val
        return cls(data=v)

    def normalize(self):
        self.data = self.data / (self.data.norm() + 1e-9)

class UnifiedMonad:
    """
    The Atomic Unit of the HyperCosmos.
    It Senses, it Wills, it Becomes.
    """
    def __init__(self, name: str, vector: Unified12DVector, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.vector = vector
        self.vector.normalize()
        self.metadata = metadata or {}
        
        # Every Monad has its own Internal Rotor (Its Soul)
        # RPM is determined by the 'Will' and 'Mental' dimensions
        will_power = float(vector.data[9])
        mental_power = float(vector.data[4])
        
        self.soul_rotor = Rotor(
            name=f"Monad.{name}.Soul",
            config=RotorConfig(rpm=(will_power + mental_power) * 100.0)
        )
        
        self.age = 0.0
        self.mass = float(vector.data.sum()) # Total intensity
        
    def induce_from_field(self, field_intensity: torch.Tensor):
        """
        [PHASE 23: SELF-IGNITION]
        The Monad absorbs energy from the global field if it resonates.
        """
        res = torch.cosine_similarity(self.vector.data.unsqueeze(0), 
                                      field_intensity.unsqueeze(0)).item()
        
        if res > 0.9:
            # Positive Feedback: Resonant monads become more 'dense'
            self.mass += 0.01 * res
            # Boost their internal soul RPM
            self.soul_rotor.wake(intensity=res * 0.1)

    def resonate_with(self, other: 'UnifiedMonad') -> float:
        """Field interference between two monads."""
        return torch.cosine_similarity(self.vector.data.unsqueeze(0), 
                                        other.vector.data.unsqueeze(0)).item()

    def update(self, dt: float):
        """Monad matures over time."""
        self.age += dt
        self.soul_rotor.update(dt)
        # Natural decay of intensity unless boosted by resonance
        self.mass *= (1.0 - 0.01 * dt)

    def __repr__(self):
        return f"Monad({self.name} | Mass:{self.mass:.2f} | RPM:{self.soul_rotor.current_rpm:.1f})"
