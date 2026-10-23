"""
Elysia Core Architecture: Embodied Cognition Schemas & DTOs

This module defines immutable data structures (DTOs) for continuous sensory vectors,
symbol prototypes, and multi-modal sensory frames.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass(frozen=True)
class ProtoVector:
    values: Tuple[float, ...]
    dimension: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'dimension', len(self.values))


@dataclass(frozen=True)
class QualitySymbol:
    name: str
    tau_threshold: float
    description: str = ""


@dataclass
class SensoryFrame:
    timestamp_ms: float
    modalities: Dict[str, List[float]]  # e.g. {"visual": [...], "tactile": [...], "motor": [...]}
    metadata: Dict[str, str] = field(default_factory=dict)
