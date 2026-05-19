from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, Set, Tuple

from .agent import Agent


@dataclass
class RectArea:
    x1: float
    y1: float
    x2: float
    y2: float

    def normalized(self) -> "RectArea":
        x_min = min(self.x1, self.x2)
        x_max = max(self.x1, self.x2)
        y_min = min(self.y1, self.y2)
        y_max = max(self.y1, self.y2)
        return RectArea(x_min, y_min, x_max, y_max)

    def intersects(self, other: Tuple[float, float, float, float]) -> bool:
        ox, oy, ow, oh = other
        rx0, ry0, rx1, ry1 = self.normalized().as_tuple()
        return not (ox + ow < rx0 or ox > rx1 or oy + oh < ry0 or oy > ry1)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class SelectionManager:
    def __init__(self, agents: Iterable[Agent]):
        self._agents: Dict[int, Agent] = {agent.id: agent for agent in agents}
        self.selected_ids: Set[int] = set()

    def select(self, rect: RectArea, mode: str = "replace") -> None:
        rect = rect.normalized()
        hits = {agent_id for agent_id, agent in self._agents.items()
                if rect.intersects(agent.get_bbox())}
        if mode == "replace":
            self.selected_ids = hits
        elif mode == "add":
            self.selected_ids |= hits
        elif mode == "remove":
            self.selected_ids -= hits
        else:
            raise ValueError(f"Unknown selection mode: {mode}")

    def clear(self) -> None:
        self.selected_ids.clear()

    def get_selected(self) -> Iterable[Agent]:
        for agent_id in self.selected_ids:
            agent = self._agents.get(agent_id)
            if agent:
                yield agent

    def get_agent(self, agent_id: int) -> Agent | None:
        return self._agents.get(agent_id)

    def update_agents(self, agents: Iterable[Agent]) -> None:
        """Sync new agent list if the world reuses objects."""
        self._agents = {agent.id: agent for agent in agents}
