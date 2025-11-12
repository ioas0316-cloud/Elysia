import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class CellMemoryStore:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path or "saves/cell_memory.json")
        self.memories: Dict[str, List[Dict[str, Any]]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.memories = {}
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                self.memories = json.load(f)
        except (json.JSONDecodeError, OSError):
            self.memories = {}

    def dump(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    def record(self, entity_id: str, event: Dict[str, Any]) -> None:
        self.memories.setdefault(entity_id, []).append(event)

    def get(self, entity_id: str) -> List[Dict[str, Any]]:
        return self.memories.get(entity_id, [])

    def clear(self) -> None:
        self.memories = {}
