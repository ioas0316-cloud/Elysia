"""
Elysia Automaton Sandbox Environment
====================================
인과 오토마톤이 지형과 장애물 필드를 자율적으로 탐색하며
기계적 맞물림과 반사를 실시간으로 현현하는 대화형 샌드박스.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from modules.causal_game_engine.causal_automaton import CausalAutomaton


class AutomatonSandbox:
    """
    오토마톤이 활동하는 물리적 인과 필드 샌드박스.
    """

    def __init__(self, width: int = 20, height: int = 15):
        self.width = width
        self.height = height
        self.obstacles: Set[Tuple[int, int]] = set()
        self.automaton: Optional[CausalAutomaton] = None
        self.trajectory_history: List[Tuple[int, int]] = []

        self._build_default_boundaries()

    def _build_default_boundaries(self):
        """외벽 경계(Boundary Condition C) 구축"""
        for x in range(self.width):
            self.obstacles.add((x, 0))
            self.obstacles.add((x, self.height - 1))
        for y in range(self.height):
            self.obstacles.add((0, y))
            self.obstacles.add((self.width - 1, y))

    def add_obstacle(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.obstacles.add((x, y))

    def add_obstacle_wall(self, start: Tuple[int, int], end: Tuple[int, int]):
        x0, y0 = start
        x1, y1 = end
        if x0 == x1:
            for y in range(min(y0, y1), max(y0, y1) + 1):
                self.add_obstacle(x0, y)
        elif y0 == y1:
            for x in range(min(x0, x1), max(x0, x1) + 1):
                self.add_obstacle(x, y0)

    def is_obstacle(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return (x, y) in self.obstacles

    def spawn_automaton(self, automaton: CausalAutomaton):
        self.automaton = automaton
        self.trajectory_history.append((automaton.x, automaton.y))

    def step(self) -> Optional[Dict[str, Any]]:
        """1 샌드박스 틱 전개"""
        if not self.automaton:
            return None

        result = self.automaton.tick(self.is_obstacle)
        self.trajectory_history.append(result["pos"])
        return result

    def render_viewport(self) -> str:
        """현재 샌드박스의 상태를 ASCII 뷰포트로 렌더링"""
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (x, y) in self.obstacles:
                    row.append("#")
                elif self.automaton and self.automaton.x == x and self.automaton.y == y:
                    row.append(result_symbol(self.automaton.dir_idx))
                elif (x, y) in self.trajectory_history[:-1]:
                    row.append(".") # 지나온 인과 궤적
                else:
                    row.append(" ")
            lines.append("".join(row))
        return "\n".join(lines)


def result_symbol(dir_idx: int) -> str:
    symbols = ["^", ">", "v", "<"]
    return symbols[dir_idx]
