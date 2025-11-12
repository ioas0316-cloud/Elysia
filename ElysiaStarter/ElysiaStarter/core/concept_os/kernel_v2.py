from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ConceptMessage:
    concept_id: str
    vector: list
    tags: List[str] = field(default_factory=list)
    priority: float = 0.5
    ttl: int = 3
    payload: Dict[str, Any] = field(default_factory=dict)

class Nanobot:
    name: str = "anon"
    subscribes: List[str] = []
    def on_message(self, msg: ConceptMessage):
        pass
    def tick(self, dt: float):
        pass

class ConceptKernelV2:
    def __init__(self):
        self.subscribers: Dict[str, List[Nanobot]] = {}
        self.time_rate: Dict[str, float] = {}

    def post(self, msg: ConceptMessage):
        for key, subs in self.subscribers.items():
            if key==msg.concept_id or key in msg.tags:
                for s in subs:
                    s.on_message(msg)
        msg.ttl -= 1
        return msg.ttl>0

    def subscribe(self, key: str, nb: Nanobot):
        self.subscribers.setdefault(key, []).append(nb)

    def tick(self, scope: str, dt: float):
        rate = self.time_rate.get(scope, 1.0)
        return dt * rate

    def set_time_rate(self, scope: str, rate: float):
        self.time_rate[scope] = rate
