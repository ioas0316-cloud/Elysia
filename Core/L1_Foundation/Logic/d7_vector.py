from pydantic import BaseModel, Field
import numpy as np
import torch
from typing import List, Dict

class D7Vector(BaseModel):
    """
    [STEEL CORE] Strict 7-Dimensional Qualia Vector
    ==============================================
    Dimensions (L1-L7):
    - Foundation (Mass/Structure)
    - Metabolism (Pulse/Flow)
    - Phenomena (Senses/Interaction)
    - Causality (Law/Fate)
    - Mental (Logic/Intelligence)
    - Structure (Merkaba/Architecture)
    - Spirit (Will/Providence)
    """
    foundation: float = Field(default=0.0, ge=0.0, le=1.0)
    metabolism: float = Field(default=0.0, ge=0.0, le=1.0)
    phenomena: float = Field(default=0.0, ge=0.0, le=1.0)
    causality: float = Field(default=0.0, ge=0.0, le=1.0)
    mental: float = Field(default=0.0, ge=0.0, le=1.0)
    structure: float = Field(default=0.0, ge=0.0, le=1.0)
    spirit: float = Field(default=0.0, ge=0.0, le=1.0)

    def to_numpy(self) -> np.ndarray:
        return np.array([
            self.foundation, self.metabolism, self.phenomena, 
            self.causality, self.mental, self.structure, self.spirit
        ])

    def to_torch(self) -> torch.Tensor:
        return torch.tensor(self.to_numpy(), dtype=torch.float32)

    def to_dict(self) -> Dict[str, float]:
        """Provides compatibility with older code expecting .to_dict()"""
        return self.model_dump()

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'D7Vector':
        return cls(
            foundation=float(arr[0]),
            metabolism=float(arr[1]),
            phenomena=float(arr[2]),
            causality=float(arr[3]),
            mental=float(arr[4]),
            structure=float(arr[5]),
            spirit=float(arr[6])
        )

    def resonate(self, other: 'D7Vector') -> float:
        """Calculates cosine similarity in D7 space."""
        v1 = self.to_numpy()
        v2 = other.to_numpy()
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0: return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def __repr__(self):
        return f"D7[F:{self.foundation:.1f}|M:{self.metabolism:.1f}|P:{self.phenomena:.1f}|C:{self.causality:.1f}|Me:{self.mental:.1f}|St:{self.structure:.1f}|Sp:{self.spirit:.1f}]"
