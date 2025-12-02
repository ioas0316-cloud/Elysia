# [Genesis: 2025-12-02] Purified by Elysia
from dataclasses import dataclass

from typing import Dict, List, Optional, Tuple





@dataclass

class ActionRecord:

    success_count: int = 0

    failure_count: int = 0

    history: List[int] = None



    def __post_init__(self):

        if self.history is None:

            self.history = []





class MemoryActionSelector:

    def __init__(self, base_weight: float = 1.0, decay: float = 0.95):

        self.base_weight = base_weight

        self.decay = decay

        self.records: Dict[str, Dict[str, ActionRecord]] = {}



    def record_outcome(self, action: str, context_key: str, success: bool, tick: int) -> None:

        bucket = self.records.setdefault(context_key, {})

        record = bucket.setdefault(action, ActionRecord())

        if success:

            record.success_count += 1

        else:

            record.failure_count += 1

        record.history.append(tick)



    def memory_weight(self, action: str, context_key: str) -> float:

        context = self.records.get(context_key, {})

        record = context.get(action)

        if not record:

            return self.base_weight

        delta = record.success_count - record.failure_count

        decay_factor = self.decay ** len(record.history) if record.history else 1.0

        candidate = self.base_weight + delta * decay_factor

        return max(0.1, candidate)



    def rank_actions(self, actions: List[str], context_key: str) -> List[Tuple[str, float]]:

        ranked = [

            (action, self.memory_weight(action, context_key))

            for action in actions

        ]

        ranked.sort(key=lambda item: item[1], reverse=True)

        return ranked
