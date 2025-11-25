from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time
import itertools


_msg_id = itertools.count(1)


@dataclass
class Message:
    verb: str
    slots: Dict[str, Any] = field(default_factory=dict)
    src: str = "user"
    dst: Optional[str] = None
    strength: float = 1.0
    ttl: int = 3
    ts: float = field(default_factory=lambda: time.time())
    id: int = field(default_factory=lambda: next(_msg_id))

    def decay(self) -> None:
        self.ttl = max(0, self.ttl - 1)
        self.strength = max(0.0, self.strength * 0.9)

