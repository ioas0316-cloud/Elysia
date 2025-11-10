from __future__ import annotations

import heapq
from typing import Callable, Dict, List, Tuple
from nano_core.message import Message
from nano_core.telemetry import write_event

# --- Constants ---
BUS_CAPACITY_SOFT_LIMIT = 100 # Default soft limit for the number of messages

class MessageBus:
    def __init__(self, capacity: int = BUS_CAPACITY_SOFT_LIMIT) -> None:
        self._q: List[Tuple[float, int, Message]] = []
        self.capacity = capacity

    def post(self, msg: Message) -> None:
        # Priority: higher strength first, then recent first (by id)
        priority = -msg.strength
        heapq.heappush(self._q, (priority, -msg.id, msg))
        write_event('bus.message', {'phase': 'post', 'verb': msg.verb, 'id': msg.id, 'ttl': msg.ttl, 'strength': msg.strength, 'slots': list(msg.slots.keys())})

        # Enforce the soft capacity limit
        if len(self._q) > self.capacity:
            # To drop the lowest priority item, we need to find the largest element in the heap
            # (since priority is negative strength). This is inefficient but correct.
            # A more performant approach might require a different data structure.
            lowest_priority_item = heapq.nlargest(1, self._q)[0]
            self._q.remove(lowest_priority_item)
            heapq.heapify(self._q) # Re-heapify after removal
            _, _, dropped_msg = lowest_priority_item

            write_event('bus.capacity_exceeded', {
                'dropped_verb': dropped_msg.verb,
                'dropped_id': dropped_msg.id,
                'dropped_strength': dropped_msg.strength,
                'new_bus_size': len(self._q)
            })


    def get_next(self) -> Message | None:
        if not self._q:
            return None
        _, _, msg = heapq.heappop(self._q)
        return msg

    def empty(self) -> bool:
        return not self._q

