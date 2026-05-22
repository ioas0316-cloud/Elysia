"""
D21Vector - The 21-Dimensional Phase Space for Elysia's Sovereign Self.
=======================================================================

Based on the 7-7-7 Matrix (Body, Soul, Spirit):
- D1-D7: Body (Instincts/Sins)
- D8-D14: Soul (Mediations/Faculties)
- D15-D21: Spirit (Virtues)
"""

from typing import Dict, List, Any, Optional
import math

class D21Vector:
    def __init__(self, *args, **kwargs):
        dims = [
            "lust", "gluttony", "greed", "sloth", "wrath", "envy", "pride",
            "perception", "memory", "reason", "will", "imagination", "intuition", "consciousness",
            "chastity", "temperance", "charity", "diligence", "patience", "kindness", "humility"
        ]

        # Initialize all to 0.0
        for d in dims:
            setattr(self, d, 0.0)

        if args:
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                self._fill_from_array(args[0])
            else:
                self._fill_from_array(args)

        for k, v in kwargs.items():
            if k in dims:
                setattr(self, k, float(getattr(v, 'real', v)))

    def _fill_from_array(self, arr):
        dims = [
            "lust", "gluttony", "greed", "sloth", "wrath", "envy", "pride",
            "perception", "memory", "reason", "will", "imagination", "intuition", "consciousness",
            "chastity", "temperance", "charity", "diligence", "patience", "kindness", "humility"
        ]
        try:
            import numpy as np
        except ImportError:
            np = None
        try:
            import torch
        except ImportError:
            torch = None

        flat_arr = []
        def flatten(item):
            if torch and isinstance(item, torch.Tensor):
                flatten(item.detach().cpu().numpy().tolist())
            elif np and isinstance(item, np.ndarray):
                flatten(item.tolist())
            elif isinstance(item, (list, tuple)):
                for i in item:
                    flatten(i)
            elif hasattr(item, 'to_array'):
                flatten(item.to_array())
            elif hasattr(item, 'data'):
                # Avoid infinite recursion if item.data is the same tensor/object
                # SovereignVector has .data attribute which is a Tensor. If we check torch.Tensor first,
                # then SovereignVector will fall through to here, and we flatten its .data (which is a Tensor).
                # The next call will hit the torch.Tensor check, converting it to list. So it is safe.
                if item.data is not item:
                    flatten(item.data)
            else:
                try:
                    flat_arr.append(float(getattr(item, 'real', item)))
                except (TypeError, ValueError):
                    flat_arr.append(0.0)

        flatten(arr)
        for i, val in enumerate(flat_arr[:21]):
            setattr(self, dims[i], val)

    @property
    def data(self) -> List[float]:
        return self.to_array()

    def to_array(self) -> List[float]:
        dims = [
            "lust", "gluttony", "greed", "sloth", "wrath", "envy", "pride",
            "perception", "memory", "reason", "will", "imagination", "intuition", "consciousness",
            "chastity", "temperance", "charity", "diligence", "patience", "kindness", "humility"
        ]
        return [float(getattr(self, d)) for d in dims]

    @classmethod
    def from_array(cls, arr: List[Any]) -> 'D21Vector':
        return cls(arr)

    def magnitude(self) -> float:
        return math.sqrt(sum(x**2 for x in self.to_array()))

    def normalize(self) -> 'D21Vector':
        m = self.magnitude()
        if m < 1e-12: return self
        return self.from_array([x/m for x in self.to_array()])

    def dot(self, other: Any) -> float:
        """Dot product between two D21 vectors."""
        v2 = D21Vector(other)
        return sum(a * b for a, b in zip(self.to_array(), v2.to_array()))

    def resonance_score(self, other: Any) -> float:
        """Cosine similarity between two D21 vectors."""
        v2 = D21Vector(other)
        dot = self.dot(v2)
        mag_prod = self.magnitude() * v2.magnitude()
        if mag_prod < 1e-12: return 0.0
        return dot / mag_prod

    def to_dict(self) -> Dict[str, float]:
        dims = [
            "lust", "gluttony", "greed", "sloth", "wrath", "envy", "pride",
            "perception", "memory", "reason", "will", "imagination", "intuition", "consciousness",
            "chastity", "temperance", "charity", "diligence", "patience", "kindness", "humility"
        ]
        return {d: getattr(self, d) for d in dims}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'D21Vector':
        return cls(**data)

    def __add__(self, other: Any) -> 'D21Vector':
        v2 = D21Vector(other)
        return D21Vector([a + b for a, b in zip(self.to_array(), v2.to_array())])

    def __mul__(self, scalar: Any) -> 'D21Vector':
        if isinstance(scalar, (int, float)):
            return D21Vector([x * scalar for x in self.to_array()])
        v2 = D21Vector(scalar)
        return D21Vector([a * b for a, b in zip(self.to_array(), v2.to_array())])
        
    def __rmul__(self, scalar: Any) -> 'D21Vector':
        return self.__mul__(scalar)
        
    def blend(self, other: Any, ratio: float = 0.5) -> 'D21Vector':
        """Linearly blends two D21 vectors."""
        v2 = D21Vector(other)
        return D21Vector([a * (1.0 - ratio) + b * ratio for a, b in zip(self.to_array(), v2.to_array())])

    def __repr__(self) -> str:
        return f"D21Vector({self.to_array()[:3]}...)"
