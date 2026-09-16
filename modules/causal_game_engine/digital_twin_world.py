"""
Elysia Digital Twin World Environment
=====================================
현실의 위상과 인과 역학(잠긴 문, 열쇠 도구, 목적지 성소)을 복제하여
주체적 관측자가 체화된 인과 지식을 탐색/학습할 수 있도록 구축된 디지털 트윈 세계.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from modules.causal_game_engine.volitional_observer import VolitionalObserver
from modules.causal_game_engine.automaton_sandbox import result_symbol


class DigitalTwinWorld:
    """
    현실의 물리적/인과적 법칙이 결상된 디지털 트윈 샌드박스.
    """

    def __init__(self, width: int = 22, height: int = 12):
        self.width = width
        self.height = height
        self.obstacles: Set[Tuple[int, int]] = set()
        self.gate_pos: Tuple[int, int] = (10, 5)
        self.gate_unlocked: bool = False
        self.shrine_pos: Tuple[int, int] = (18, 5)
        self.tools: Dict[Tuple[int, int], Dict[str, Any]] = {
            (5, 8): {"id": "key_card", "name": "Golden_Key"}
        }
        self.observer: Optional[VolitionalObserver] = None
        self.history: List[Tuple[int, int]] = []

        self._build_environment()

    def _build_environment() -> None:
        pass  # will implement in instance

    def _build_environment(self):
        # 외벽
        for x in range(self.width):
            self.obstacles.add((x, 0))
            self.obstacles.add((x, self.height - 1))
        for y in range(self.height):
            self.obstacles.add((0, y))
            self.obstacles.add((self.width - 1, y))

        # 중앙 격벽 (y=1부터 10까지, y=5에 관문 G 배치)
        for y in range(1, self.height - 1):
            if y != self.gate_pos[1]:
                self.obstacles.add((self.gate_pos[0], y))

    def is_obstacle(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        if (x, y) in self.obstacles:
            return True
        # 잠긴 문일 경우 장애물로 판정
        if (x, y) == self.gate_pos and not self.gate_unlocked:
            return True
        return False

    def spawn_observer(self, observer: VolitionalObserver):
        self.observer = observer
        self.history.append((observer.x, observer.y))

    def step(self) -> Optional[Dict[str, Any]]:
        if not self.observer:
            return None

        res = self.observer.volitional_step(
            world_query_fn=self.is_obstacle,
            tools_in_world=self.tools,
            gate_pos=self.gate_pos if not self.gate_unlocked else None
        )

        # 관문 해제 이벤트 동기화
        if "UNLOCK_GATE" in res["action"]:
            self.gate_unlocked = True

        self.history.append((self.observer.x, self.observer.y))
        return res

    def render_viewport(self) -> str:
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)
                if pos in self.obstacles:
                    row.append("#")
                elif pos == self.gate_pos:
                    row.append(" " if self.gate_unlocked else "G")
                elif pos in self.tools:
                    row.append("K")
                elif pos == self.shrine_pos:
                    row.append("S")
                elif self.observer and (self.observer.x, self.observer.y) == pos:
                    row.append(result_symbol(self.observer.dir_idx))
                elif pos in self.history[:-1]:
                    row.append(".")
                else:
                    row.append(" ")
            lines.append("".join(row))
        return "\n".join(lines)
