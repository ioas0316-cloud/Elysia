"""
Capability Registry - tracks self-assessed competencies and deficits.
Lightweight store used by Observer/Dreamer to plan self-improvement.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("CapabilityRegistry")


@dataclass
class CapabilityRecord:
    name: str
    score: float = 0.5  # 0.0~1.0
    status: str = "unknown"
    notes: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ImprovementTicket:
    ticket_id: str
    target: str
    issue: str
    suggestion: str
    status: str = "open"  # open, in_progress, done, rejected


class CapabilityRegistry:
    def __init__(self):
        self.capabilities: Dict[str, CapabilityRecord] = {
            "language": CapabilityRecord("language", score=0.7, status="stable"),
            "memory": CapabilityRecord("memory", score=0.6, status="growing"),
            "phase": CapabilityRecord("phase", score=0.6, status="growing"),
            "defense": CapabilityRecord("defense", score=0.3, status="weak"),
            "planning": CapabilityRecord("planning", score=0.5, status="growing"),
        }
        self.tickets: Dict[str, ImprovementTicket] = {}

    def update(self, name: str, score: float, status: str, notes: str = "", tags: Optional[List[str]] = None):
        rec = self.capabilities.get(name) or CapabilityRecord(name)
        rec.score = max(0.0, min(1.0, score))
        rec.status = status
        rec.notes = notes
        if tags:
            rec.tags = list(set(rec.tags + tags))
        self.capabilities[name] = rec

    def assess_from_snapshot(self, snapshot: Dict) -> List[CapabilityRecord]:
        """Basic heuristics to auto-update some capabilities from runtime snapshot."""
        reports: List[CapabilityRecord] = []
        phase = snapshot.get("phase", {})
        q = phase.get("quaternion", {})
        mastery = abs(q.get("w", 0.0))
        if mastery < 0.3:
            self.update("phase", score=mastery, status="drifting", notes="low mastery")
            reports.append(self.capabilities["phase"])

        mem = snapshot.get("memory", {})
        edge_count = mem.get("causal_edges", 0)
        if edge_count > 1000:
            self.update("memory", score=min(1.0, 0.7 + edge_count / 5000.0), status="expanding")
        else:
            self.update("memory", score=0.6, status="growing")

        # Defense placeholder: if chaos spike recorded, mark weak
        chaos = abs(snapshot.get("chaos_raw", 0.0))
        if chaos > 50:
            self.update("defense", score=0.2, status="weak", notes="chaos spike observed")
            reports.append(self.capabilities["defense"])

        return reports

    def deficits(self, threshold: float = 0.5) -> List[CapabilityRecord]:
        """Return capabilities below threshold."""
        return [rec for rec in self.capabilities.values() if rec.score < threshold]

    # === Improvement tickets ===
    def has_open_ticket(self, target: str) -> bool:
        return any(t for t in self.tickets.values() if t.target == target and t.status == "open")

    def add_ticket(self, target: str, issue: str, suggestion: str) -> ImprovementTicket:
        if self.has_open_ticket(target):
            # Return existing open ticket to avoid spam
            return [t for t in self.tickets.values() if t.target == target and t.status == "open"][0]
        ticket_id = f"ticket_{len(self.tickets)+1:04d}"
        ticket = ImprovementTicket(ticket_id, target, issue, suggestion)
        self.tickets[ticket_id] = ticket
        return ticket

    def list_open_tickets(self) -> List[ImprovementTicket]:
        return [t for t in self.tickets.values() if t.status == "open"]

    def resolve_ticket(self, ticket_id: str, status: str = "done"):
        if ticket_id in self.tickets:
            self.tickets[ticket_id].status = status
