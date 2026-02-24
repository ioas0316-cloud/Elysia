"""Causal admissibility gate implementation.

Implements the first constraint from the causal blueprint:
admissible = has_cause AND phase_coherent AND energy_safe AND will_aligned AND trinary_stable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence
from threading import RLock
import math


@dataclass(frozen=True)
class CausalSignature:
    """Evidence bundle used to evaluate a state transition."""

    cause_id: Optional[str]
    intent_vector: Sequence[float]
    result_vector: Sequence[float]
    phase_delta: float
    energy_cost: float
    trinary_state: Dict[str, float]
    context_hash: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class GateThresholds:
    """Operating thresholds for CAG."""

    max_phase_delta: float = math.pi / 2
    max_energy_cost: float = 1.0
    min_will_alignment: float = 0.7
    min_neutral_ratio: float = 0.10
    min_trinary_sum: float = 0.99
    max_trinary_sum: float = 1.01


@dataclass
class TransitionRecord:
    """Audit record for every transition decision."""

    from_state: str
    to_state: str
    signature: CausalSignature
    admissible: bool
    rejection_reasons: List[str]
    resonance_score: float
    audit_tag: str = "CAG_V1"


class CausalAdmissibilityGate:
    """Constraint gate that decides whether a transition is admissible."""

    def __init__(self, thresholds: Optional[GateThresholds] = None):
        self.thresholds = thresholds or GateThresholds()
        self.ledger: List[TransitionRecord] = []
        self.quarantine: List[TransitionRecord] = []
        self._lock = RLock()

    def evaluate(
        self,
        *,
        from_state: str,
        to_state: str,
        signature: CausalSignature,
        resonance_score: float = 0.0,
    ) -> TransitionRecord:
        """Evaluate a transition and store the result in the ledger."""
        reasons: List[str] = []

        if not self._has_cause(signature):
            reasons.append("missing_cause")
        if not self._phase_coherent(signature):
            reasons.append("phase_incoherent")
        if not self._energy_safe(signature):
            reasons.append("energy_over_budget")
        if not self._will_aligned(signature):
            reasons.append("will_misaligned")
        if not self._trinary_stable(signature):
            reasons.append("trinary_unstable")

        admissible = len(reasons) == 0
        record = TransitionRecord(
            from_state=from_state,
            to_state=to_state,
            signature=signature,
            admissible=admissible,
            rejection_reasons=reasons,
            resonance_score=resonance_score,
        )

        with self._lock:
            self.ledger.append(record)
            if not admissible:
                self.quarantine.append(record)

        return record

    def _has_cause(self, signature: CausalSignature) -> bool:
        return bool(signature.cause_id and signature.cause_id.strip())

    def _phase_coherent(self, signature: CausalSignature) -> bool:
        return abs(signature.phase_delta) <= self.thresholds.max_phase_delta

    def _energy_safe(self, signature: CausalSignature) -> bool:
        return signature.energy_cost <= self.thresholds.max_energy_cost

    def _will_aligned(self, signature: CausalSignature) -> bool:
        similarity = cosine_similarity(signature.intent_vector, signature.result_vector)
        return similarity >= self.thresholds.min_will_alignment

    def _trinary_stable(self, signature: CausalSignature) -> bool:
        neg = float(signature.trinary_state.get("negative", 0.0))
        neu = float(signature.trinary_state.get("neutral", 0.0))
        pos = float(signature.trinary_state.get("positive", 0.0))
        total = neg + neu + pos

        if not (self.thresholds.min_trinary_sum <= total <= self.thresholds.max_trinary_sum):
            return False

        if neu < self.thresholds.min_neutral_ratio:
            return False

        return min(neg, neu, pos) >= 0.0

    def quarantine_ratio(self) -> float:
        """Return ratio of quarantined transitions across all evaluated transitions."""
        with self._lock:
            if not self.ledger:
                return 0.0
            return len(self.quarantine) / len(self.ledger)

    def drain_quarantine(self, limit: Optional[int] = None) -> List[TransitionRecord]:
        """Drain quarantined transitions in FIFO order for recovery workers."""
        with self._lock:
            if limit is None or limit >= len(self.quarantine):
                drained = list(self.quarantine)
                self.quarantine.clear()
                return drained

            drained = self.quarantine[:limit]
            del self.quarantine[:limit]
            return drained

    def snapshot(self) -> Dict[str, float]:
        """Thread-safe operational metrics for supervising loops."""
        with self._lock:
            ledger_count = len(self.ledger)
            quarantine_count = len(self.quarantine)

        return {
            "ledger_count": float(ledger_count),
            "quarantine_count": float(quarantine_count),
            "quarantine_ratio": (quarantine_count / ledger_count) if ledger_count else 0.0,
        }


class ResonanceLedger:
    """Simple append-only ledger helper for querying stored records."""

    def __init__(self, records: Optional[List[TransitionRecord]] = None):
        self.records: List[TransitionRecord] = records or []

    def append(self, record: TransitionRecord) -> None:
        self.records.append(record)

    def rejected(self) -> List[TransitionRecord]:
        return [r for r in self.records if not r.admissible]

    def by_cause(self, cause_id: str) -> List[TransitionRecord]:
        return [r for r in self.records if r.signature.cause_id == cause_id]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity safely for potentially empty vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)
