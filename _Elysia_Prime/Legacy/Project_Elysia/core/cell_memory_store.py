# [Genesis: 2025-12-02] Purified by Elysia
import json

from collections import deque

from pathlib import Path

from typing import Any, Deque, Dict, List, Optional





class CellMemoryStore:

    """

    Lightweight fractal memory layer for cells.



    Each cell id maps to a ring buffer (deque) of recent events, so memory

    growth is bounded even in long-running worlds. This mirrors the ring-based

    design of Elysia's CoreMemory at a simpler, per-cell level.

    """



    def __init__(self, path: Optional[str] = None, capacity: int = 200) -> None:

        self.path = Path(path or "saves/cell_memory.json")

        self.capacity = int(capacity)

        self.memories: Dict[str, Deque[Dict[str, Any]]] = {}

        self.load()



    def load(self) -> None:

        if not self.path.exists():

            self.memories = {}

            return

        try:

            with self.path.open("r", encoding="utf-8") as f:

                raw = json.load(f)

            memories: Dict[str, Deque[Dict[str, Any]]] = {}

            for key, value in raw.items():

                if isinstance(value, list):

                    items = value

                elif value:

                    items = [value]

                else:

                    items = []

                memories[key] = deque(items, maxlen=self.capacity)

            self.memories = memories

        except (json.JSONDecodeError, OSError):

            self.memories = {}



    def dump(self) -> None:

        self.path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {key: list(events) for key, events in self.memories.items()}

        with self.path.open("w", encoding="utf-8") as f:

            json.dump(serializable, f, ensure_ascii=False, indent=2)



    def record(self, entity_id: str, event: Dict[str, Any]) -> None:

        if entity_id not in self.memories:

            self.memories[entity_id] = deque(maxlen=self.capacity)

        self.memories[entity_id].append(event)



    def get(self, entity_id: str) -> List[Dict[str, Any]]:

        return list(self.memories.get(entity_id, []))



    def clear(self) -> None:

        self.memories = {}
