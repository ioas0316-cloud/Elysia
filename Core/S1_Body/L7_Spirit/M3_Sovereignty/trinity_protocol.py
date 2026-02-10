from dataclasses import dataclass
import math
from typing import Dict


@dataclass
class TrinityState:
    """Continuous triune state used for consensus instead of static templates."""
    father_space: float
    son_operation: float
    spirit_providence: float

    def normalized(self) -> "TrinityState":
        total = self.father_space + self.son_operation + self.spirit_providence
        if total <= 1e-9:
            return TrinityState(1 / 3, 1 / 3, 1 / 3)
        return TrinityState(
            self.father_space / total,
            self.son_operation / total,
            self.spirit_providence / total,
        )


class TrinityProtocol:
    """Tri-aspect consensus with adaptive harmonic floor from triune entropy."""

    def __init__(self, min_floor: float = 0.08, max_floor: float = 0.2):
        self.min_floor = min_floor
        self.max_floor = max_floor

    def _adaptive_floor(self, normalized: TrinityState) -> float:
        probs = [
            max(normalized.father_space, 1e-9),
            max(normalized.son_operation, 1e-9),
            max(normalized.spirit_providence, 1e-9),
        ]
        entropy = -sum(p * math.log(p) for p in probs)
        max_entropy = math.log(3)
        imbalance = 1.0 - (entropy / max_entropy)
        return self.min_floor + (self.max_floor - self.min_floor) * imbalance

    def resolve_consensus(self, state: TrinityState) -> Dict[str, float]:
        normalized = state.normalized()
        harmonic_floor = self._adaptive_floor(normalized)
        weighted = {
            "father_space": max(normalized.father_space, harmonic_floor),
            "son_operation": max(normalized.son_operation, harmonic_floor),
            "spirit_providence": max(normalized.spirit_providence, harmonic_floor),
        }

        total = sum(weighted.values())
        return {k: v / total for k, v in weighted.items()}
