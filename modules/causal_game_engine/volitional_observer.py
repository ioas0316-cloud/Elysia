"""
Elysia Volitional Observer Core Module
======================================
SOMA(신체), LOGOS(도구와 렌즈), NOUS(의지와 목적)의 삼위일체 결합체.
수동적 확률 가중치가 아닌, 내재된 텔레올로지(목적과 결핍)와 환경의 가용성(Affordance)이
공명하여 도구를 선택하고 신체화하며, 현실의 제약을 풀어내는 '주체적 관측자'.

핵심 원리:
- Query (NOUS): "목적지까지 가로막힌 인과적 장력(Tension)은 무엇인가?"
- Key (LOGOS): "환경 내 존재하는 도구(Key, Lever) 중 제약 C를 해제할 열쇠는 무엇인가?"
- Value (SOMA): "도구를 신체화하고 모멘텀을 집중하여 제약을 해소한다."
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import math

from core.topology.causal_stem_branch_engine import (
    CausalGraph, CausalNode, CausalEdge, NodeType, TrajectoryContext
)
from modules.causal_game_engine.causal_automaton import (
    CausalAutomaton,
    DIRECTIONS,
    DIR_NAMES,
    DIR_SYMBOLS
)

# 도구 및 관문 불변 서명
INVARIANT_TOOL_KEY = "INVARIANT_TOOL_KEY"
INVARIANT_GATE_LOCKED = "INVARIANT_GATE_LOCKED"
INVARIANT_TELEOLOGY_SHRINE = "INVARIANT_TELEOLOGY_SHRINE"


class CausalAttentionEngine:
    """
    주체적 의지와 환경 가용성(Affordance)을 매칭하는 인과적 어텐션 엔진.
    """

    def resolve_attention(
        self,
        current_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        is_blocked_by_gate: bool,
        visible_tools: List[Dict[str, Any]],
        has_tool: bool
    ) -> str:
        """
        의지적 결단(Volitional Decision):
        - 문이 막혀있지 않거나 이미 열쇠가 있으면 -> 목표(Goal)로 어텐션 집중
        - 문이 잠겨있고 열쇠가 없으면 -> 열쇠 도구(Tool)로 어텐션 전환(Lens Shift)
        """
        if not is_blocked_by_gate and has_tool:
            return "ATTEND_TO_GOAL"
        elif is_blocked_by_gate and not has_tool:
            if visible_tools:
                return "ATTEND_TO_TOOL"
            else:
                return "EXPLORE_FIELD"
        else:
            return "ATTEND_TO_GOAL"


class VolitionalObserver(CausalAutomaton):
    """
    목적과 자아, 도구 신체화 능력을 지닌 삼위일체 주체적 관측자.
    """

    def __init__(
        self,
        observer_id: str = "volitional_ego_1",
        x: int = 2,
        y: int = 2,
        goal_x: int = 18,
        goal_y: int = 8,
        dir_idx: int = 1
    ):
        super().__init__(automaton_id=observer_id, x=x, y=y, dir_idx=dir_idx)
        self.goal_pos = (goal_x, goal_y)
        self.attention_engine = CausalAttentionEngine()
        
        # 인지 및 신체화 상태
        self.current_lens: str = "Spatial_Movement" # "Spatial_Movement" vs "Tool_Affordance"
        self.assimilated_tools: Dict[str, CausalNode] = {}
        self.epistemic_trajectory: List[Dict[str, Any]] = []
        self.mission_completed: bool = False
        self.tension_level: float = 1.0 # 목적지와의 인과적 장력

    def calculate_tension(self) -> float:
        """목표점과의 맨해튼 거리 기반 인과적 긴장도(Tension Ω)"""
        dist = abs(self.x - self.goal_pos[0]) + abs(self.y - self.goal_pos[1])
        self.tension_level = float(dist)
        return self.tension_level

    def assimilate_tool(self, tool_id: str, tool_node: CausalNode):
        """
        도구의 신체화 (Tool Assimilation):
        외부 객체를 자신의 인과 그래프 하위 노드로 결합하여 가용 자유도(DOF)를 확장.
        """
        self.assimilated_tools[tool_id] = tool_node
        self.graph.add_node(tool_node)

        # 신체 손바닥 노드(바퀴/팔)와 도구 간 ConnectivityBeam 엣지 연결
        self.graph.add_edge(CausalEdge(
            source_id=f"{self.automaton_id}_wheel",
            target_id=tool_node.node_id,
            precondition="tool_teleological_coupling",
            is_necessary=True
        ))
        self.current_lens = "Constraint_Unlocker"

    def volitional_step(
        self,
        world_query_fn: Any,
        tools_in_world: Dict[Tuple[int, int], Dict[str, Any]],
        gate_pos: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        주체적 의지(Volition)에 의한 1사이클 행동 결단.
        """
        self.calculate_tension()
        current_pos = (self.x, self.y)

        # 1. 주변 도구 탐지
        visible_tools = []
        for t_pos, t_info in tools_in_world.items():
            dist = abs(self.x - t_pos[0]) + abs(self.y - t_pos[1])
            if dist <= 12: # 감각 인지 반경
                visible_tools.append({"pos": t_pos, "info": t_info})

        has_key = "key_card" in self.assimilated_tools
        is_blocked_by_gate = (gate_pos is not None) and not has_key

        # 2. 인과적 어텐션 발동 (의도 Q x 가용성 K -> 행동 결정)
        attention_state = self.attention_engine.resolve_attention(
            current_pos=current_pos,
            goal_pos=self.goal_pos,
            is_blocked_by_gate=is_blocked_by_gate,
            visible_tools=visible_tools,
            has_tool=has_key
        )

        # 3. 행동 목표 타겟 좌표 결정
        if attention_state == "ATTEND_TO_TOOL" and visible_tools:
            target_coord = visible_tools[0]["pos"]
            mode_desc = "TARGET_TOOL_KEY"
        else:
            target_coord = self.goal_pos
            mode_desc = "TARGET_TELEOLOGY_SHRINE"

        # 4. 타겟을 향한 인과적 벡터 산출 (단순 충돌 반사가 아닌 지향성 이동)
        best_dir = self._calculate_intentional_heading(target_coord, world_query_fn)
        self.dir_idx = best_dir

        # 전진 실행
        dx, dy = self.forward_vector
        nx, ny = self.x + dx, self.y + dy

        # 관문 도달 시 도구로 잠금 해제
        if gate_pos and (nx, ny) == gate_pos:
            if has_key:
                action = "UNLOCK_GATE_WITH_ASSIMILATED_KEY"
                self.x = nx
                self.y = ny
            else:
                action = "BLOCKED_BY_LOCKED_GATE"
        elif not world_query_fn(nx, ny):
            self.x = nx
            self.y = ny
            action = f"INTENTIONAL_ADVANCE_TO_({self.x},{self.y})"
        else:
            action = "TACTILE_OBSTACLE_AVOIDANCE"
            # 장애물이면 우회 회전
            self.dir_idx = (self.dir_idx + 1) % 4

        # 도구 위에 도착하면 자동 신체화
        current_pos = (self.x, self.y)
        if current_pos in tools_in_world:
            t_info = tools_in_world.pop(current_pos)
            tool_node = CausalNode(
                node_id=t_info["id"],
                node_type=NodeType.STEM,
                invariant_signature=INVARIANT_TOOL_KEY,
                payload={"name": t_info["name"]}
            )
            self.assimilate_tool(t_info["id"], tool_node)
            action += f"_AND_ASSIMILATE_{t_info['name']}"

        # 목표점 도달 검증
        if current_pos == self.goal_pos:
            self.mission_completed = True
            action = "REACH_TELEOLOGICAL_SHRINE_SUCCESS"

        record = {
            "pos": (self.x, self.y),
            "dir": DIR_NAMES[self.dir_idx],
            "attention": mode_desc,
            "has_tool": has_key,
            "tension": self.tension_level,
            "action": action
        }
        self.epistemic_trajectory.append(record)
        return record

    def _calculate_intentional_heading(self, target: Tuple[int, int], obstacle_check_fn) -> int:
        """목표를 향해 가장 거리가 줄어들며 장애물이 없는 최적 인과 방향 산출"""
        tx, ty = target
        best_dir = self.dir_idx
        min_dist = 999999

        for d_idx, (dx, dy) in enumerate(DIRECTIONS):
            nx, ny = self.x + dx, self.y + dy
            if not obstacle_check_fn(nx, ny):
                dist = abs(nx - tx) + abs(ny - ty)
                if dist < min_dist:
                    min_dist = dist
                    best_dir = d_idx

        return best_dir
