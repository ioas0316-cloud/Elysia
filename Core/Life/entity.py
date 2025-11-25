"""
Core/Life/entity.py

The foundational concept of a living being within the new architecture.
An entity is a container for state, capabilities, and a unique identity.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List
import uuid

@dataclass
class Resource:
    """Represents a single resource like HP, Mana, etc."""
    current: float
    max: float

    def __post_init__(self):
        self.current = float(self.current)
        self.max = float(self.max)

    def add(self, amount: float):
        """Adds an amount to the current value, clamping at max."""
        self.current = min(self.max, self.current + amount)

    def subtract(self, amount: float):
        """Subtracts an amount from the current value, clamping at 0."""
        self.current = max(0, self.current - amount)

    @property
    def ratio(self) -> float:
        """Returns the current resource ratio (0.0 to 1.0)."""
        if self.max == 0:
            return 0.0
        return self.current / self.max

@dataclass
class LivingEntity:
    """
    Represents a single living being in the world.
    This class is designed to be a data container, with logic handled by Systems.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = "entity"

    # Core resources
    hp: Resource = field(default_factory=lambda: Resource(100.0, 100.0))
    hunger: Resource = field(default_factory=lambda: Resource(100.0, 100.0))
    hydration: Resource = field(default_factory=lambda: Resource(100.0, 100.0))

    # Age and lifespan
    age: int = 0  # in ticks
    max_age: int = 1000 # in ticks

    # Other potential resources and stats can be added here
    # Example: mana: Resource = field(default_factory=lambda: Resource(50.0, 50.0))

    # A generic dictionary for additional properties and states
    properties: Dict[str, Any] = field(default_factory=dict)

    # List of capabilities or components attached to this entity
    components: List[str] = field(default_factory=list)

    @property
    def is_alive(self) -> bool:
        """Determines if the entity is alive based on HP."""
        return self.hp.current > 0
