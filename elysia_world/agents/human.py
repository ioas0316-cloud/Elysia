"""Simple agent archetype wrappers."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

from ..cell import Cell


@dataclass
class HumanAgent:
    concept_id: str
    label: str = "human"
    base_stats: Dict[str, Any] | None = None

    def spawn(self, world) -> Cell:
        payload = {
            "label": self.label,
            "element_type": "animal",
        }
        if self.base_stats:
            payload.update(self.base_stats)
        world.add_cell(self.concept_id, properties=payload)
        return world.materialize_cell(self.concept_id)

__all__ = ["HumanAgent"]
