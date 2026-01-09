"""Cell representation used by the simulation world."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class Cell:
    """Lightweight cell with optional DNA and social links."""

    id: str
    dna: Dict[str, Any] | None = None
    properties: Dict[str, Any] = field(default_factory=dict)
    energy: float = 100.0
    hp: float = 100.0
    age: int = 0
    connections: List[Tuple[str, str, float]] = field(default_factory=list)

    @property
    def is_alive(self) -> bool:
        return self.energy > 0 and self.hp > 0

    def connect(self, other: "Cell", relationship_type: str = "related_to", strength: float = 1.0):
        self.connections.append((other.id, relationship_type, float(strength)))
        return self.connections[-1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "dna": self.dna or {},
            "properties": dict(self.properties),
            "energy": float(self.energy),
            "hp": float(self.hp),
            "age": int(self.age),
        }

    def __repr__(self) -> str:
        return f"Cell(id={self.id}, energy={self.energy:.2f}, hp={self.hp:.2f}, age={self.age})"


def cell_unit(func):
    """
    [@CELL Synchronizer]
    Decorator that links a function to the ResonantField.
    When the function is called, it creates a ripple in the high-dimensional field.
    """
    from Core.Foundation.Wave.resonant_field import resonant_field
    from Core.Foundation.hyper_quaternion import Quaternion
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Execute the actual logic
        result = func(*args, **kwargs)
        
        # 2. Resonate with the Field
        # We generate a unique intent based on the function name and result
        context_hash = hash(func.__name__) % 100 / 100.0
        intent = Quaternion(1.0, context_hash, 0.5, 0.2)
        
        # Project into a random or relevant sector of the field
        # In a more advanced version, this would be based on semantic mapping
        import random
        rx, ry = random.randint(0, resonant_field.size-1), random.randint(0, resonant_field.size-1)
        resonant_field.project_intent(rx, ry, intent)
        
        # Trigger an evolution cycle
        resonant_field.evolve(dt=0.05)
        
        return result
    return wrapper

__all__ = ["Cell", "cell_unit"]
