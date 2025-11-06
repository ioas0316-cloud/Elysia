from __future__ import annotations

import heapq
from typing import Callable, Dict, List, Tuple
from nano_core.message import Message
from nano_core.telemetry import write_event


class MessageBus:
    def __init__(self) -> None:
        self._q: List[Tuple[float, int, Message]] = []

    def post(self, msg: Message) -> None:
        # Priority: higher strength first, then recent first (by id)
        priority = -msg.strength
        heapq.heappush(self._q, (priority, -msg.id, msg))
        write_event('bus.message', {'phase': 'post', 'verb': msg.verb, 'id': msg.id, 'ttl': msg.ttl, 'strength': msg.strength, 'slots': list(msg.slots.keys())})

    def get_next(self) -> Message | None:
        if not self._q:
            return None
        _, _, msg = heapq.heappop(self._q)
        return msg

    def empty(self) -> bool:
        return not self._q

