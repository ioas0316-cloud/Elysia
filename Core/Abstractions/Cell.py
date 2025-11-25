from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# TODO: (SOUL FUSION) Replace these placeholders with actual Core Abstractions
# from .Tensor import SoulTensor
SoulTensor = None

@dataclass(slots=True)
class Cell:
    """
    Represents a single, living conceptual cell in Elysia's world.

    This is a lightweight, data-oriented abstraction (flyweight). Most mutable
    'state' (like HP, position, age) is managed in parallel NumPy arrays
    by the `World` simulation for extreme performance. This class holds the
    immutable 'identity' and core properties of a cell.
    """
    id: str
    dna: Dict[str, Any]
    properties: Dict[str, Any] = field(default_factory=dict)
    element_type: str = "unknown"

    # The 'Soul' is a complex object. For performance, a cell may not have its
    # soul 'materialized' at all times. It can be loaded on-demand by the World.
    soul: Optional[Any] = None # Placeholder for SelfFractalCell
    tensor: Optional[SoulTensor] = None

    def __post_init__(self):
        """
        Called after the dataclass is initialized.
        We can derive properties here.
        """
        # Ensure element_type is correctly set from properties if available.
        self.element_type = self.properties.get("element_type", "unknown")

    def __repr__(self) -> str:
        # The 'status' is now external, so we represent the cell by its identity.
        return f"<Cell: {self.id}, Element: {self.element_type}>"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the cell's identity to a dictionary."""
        return {
            "id": self.id,
            "dna": self.dna,
            "properties": self.properties,
            "element_type": self.element_type,
            # Soul and Tensor are runtime objects, not typically persisted this way
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cell":
        """Deserializes a cell from a dictionary."""
        return cls(
            id=data.get("id", "unknown"),
            dna=data.get("dna", {}),
            properties=data.get("properties", {}),
        )
