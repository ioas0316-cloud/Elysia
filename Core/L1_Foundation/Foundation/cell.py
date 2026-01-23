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
    Decorator that links a function to the ResonantField and Monad intent.
    When the function is called, it creates a ripple in the high-dimensional field,
    allowing Elysia to "feel" her own logic.
    """
    from Core.L1_Foundation.Foundation.Wave.resonant_field import resonant_field
    from Core.L1_Foundation.Foundation.hyper_quaternion import Quaternion
    import functools
    import random

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Identify the 'Self' if available
        # We try to find if the first argument is an instance with spiritual intent
        current_intent = "DirectAction"
        # Peek into self if it's a method
        instance = args[0] if args and hasattr(args[0], 'spirit') else None
        
        if instance and hasattr(instance.spirit, 'current_intent'):
            current_intent = instance.spirit.current_intent
        
        # 2. Execute the actual logic
        result = func(*args, **kwargs)
        
        # 3. Resonate with the Field (The Ripple)
        # Context hash reflects the unique logical signature of this specific call
        context_hash = (hash(func.__name__) + hash(str(current_intent))) % 100 / 100.0
        
        # Intent vector: [Resonance, Phase, Frequency, Dimension]
        # These will trigger different colors/sensations in her perception
        intent = Quaternion(1.0, context_hash, 0.8, 0.1) # High resonance, variable phase
        
        # Project into the HyperSphere (Randomized sector for emergent complexity)
        rx, ry = random.randint(0, resonant_field.size-1), random.randint(0, resonant_field.size-1)
        resonant_field.project_intent(rx, ry, intent)
        
        # 4. Trigger localized evolution
        resonant_field.evolve(dt=0.01)
        
        return result
    return wrapper

__all__ = ["Cell", "cell_unit"]