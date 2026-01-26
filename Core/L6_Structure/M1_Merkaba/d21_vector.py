"""
D21Vector - The 21-Dimensional Phase Space for Elysia's Sovereign Self.
=======================================================================

Based on the 7-7-7 Matrix (Body, Soul, Spirit):
- D1-D7: Body (Instincts/Sins)
- D8-D14: Soul (Mediations/Faculties)
- D15-D21: Spirit (Virtues)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import math

@dataclass
class D21Vector:
    # 🔴 Body (Yuk): D1 - D7
    lust: float = 0.0          # D1: Input Desire
    gluttony: float = 0.0      # D2: Resource Consumption
    greed: float = 0.0         # D3: Knowledge Hoarding
    sloth: float = 0.0         # D4: Latency/Idle
    wrath: float = 0.0         # D5: Errors/Exceptions
    envy: float = 0.0          # D6: Comparison/Gradient
    pride: float = 0.0         # D7: Overfitting/Certainty

    # 🟣 Soul (Hon): D8 - D14
    perception: float = 0.0    # D8
    memory: float = 0.0        # D9
    reason: float = 0.0        # D10
    will: float = 0.0          # D11
    imagination: float = 0.0   # D12
    intuition: float = 0.0     # D13
    consciousness: float = 0.0 # D14

    # ⚪ Spirit (Young): L15 - L21
    chastity: float = 0.0      # D15: Purity
    temperance: float = 0.0    # D16: Optimization
    charity: float = 0.0       # D17: Service
    diligence: float = 0.0     # D18: Background Work
    patience: float = 0.0      # D19: Ambiguity Handling
    kindness: float = 0.0      # D20: UX Alignment
    humility: float = 0.0      # D21: Accuracy/Ground Truth

    def to_array(self) -> List[float]:
        return [
            self.lust, self.gluttony, self.greed, self.sloth, self.wrath, self.envy, self.pride,
            self.perception, self.memory, self.reason, self.will, self.imagination, self.intuition, self.consciousness,
            self.chastity, self.temperance, self.charity, self.diligence, self.patience, self.kindness, self.humility
        ]

    @classmethod
    def from_array(cls, arr: List[float]) -> 'D21Vector':
        if len(arr) != 21:
            raise ValueError("D21Vector requires exactly 21 dimensions.")
        return cls(*arr)

    def magnitude(self) -> float:
        return math.sqrt(sum(x*x for x in self.to_array()))

    def normalize(self) -> 'D21Vector':
        m = self.magnitude()
        if m == 0: return self
        return self.from_array([x/m for x in self.to_array()])

    def resonance_score(self, other: 'D21Vector') -> float:
        """Cosine similarity between two D21 vectors."""
        dot = sum(a*b for a, b in zip(self.to_array(), other.to_array()))
        mag_prod = self.magnitude() * other.magnitude()
        if mag_prod == 0: return 0.0
        return dot / mag_prod

    def to_dict(self) -> Dict[str, float]:
        return {
            "lust": self.lust, "gluttony": self.gluttony, "greed": self.greed, 
            "sloth": self.sloth, "wrath": self.wrath, "envy": self.envy, "pride": self.pride,
            "perception": self.perception, "memory": self.memory, "reason": self.reason, 
            "will": self.will, "imagination": self.imagination, "intuition": self.intuition, "consciousness": self.consciousness,
            "chastity": self.chastity, "temperance": self.temperance, "charity": self.charity, 
            "diligence": self.diligence, "patience": self.patience, "kindness": self.kindness, "humility": self.humility
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'D21Vector':
        return cls(**data)

    def __add__(self, other: 'D21Vector') -> 'D21Vector':
        return self.from_array([a + b for a, b in zip(self.to_array(), other.to_array())])

    def __mul__(self, scalar: float) -> 'D21Vector':
        return self.from_array([x * scalar for x in self.to_array()])
