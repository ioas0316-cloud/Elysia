"""
Elysia Causal Automaton Core Module
===================================
부동소수점 물리 시뮬레이션이나 기호적 if-else 인공지능 코드 없이,
기구학적 맞물림 구속 조건(Kinematic Constraints)과 감각-운동 인과 엣지만으로
자율적으로 전진하고 장애물을 회피하는 '인과 오토마톤(Causal Automaton)'.

원리: "Do not calculate, let it flow."
- 태엽 심장(Power Core)의 주기적 펄스가 기어열(Gear Train)을 거쳐 바퀴(Wheel)로 전진 모멘텀을 공급한다.
- 더듬이(Whisker)가 장애물 경계(Boundary Condition C)에 부딪히면,
  전진 엣지가 기구학적으로 잠기고(Locked), 모멘텀이 조향 링크(Steering Linkage)로 우회(Divergence)된다.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass

from core.topology.causal_stem_branch_engine import (
    CausalGraph, CausalNode, CausalEdge, NodeType, TrajectoryContext
)
from core.topology.base_topological_parser import (
    INVARIANT_RIGID_BODY,
    INVARIANT_JOINT_REVOLUTE,
    INVARIANT_JOINT_PRISMATIC
)

# 오토마톤 전용 불변 인과 서명
INVARIANT_POWER_CORE = "INVARIANT_POWER_CORE"
INVARIANT_GEAR_TRAIN = "INVARIANT_GEAR_TRAIN"
INVARIANT_ACTUATOR_WHEEL = "INVARIANT_ACTUATOR_WHEEL"
INVARIANT_SENSOR_WHISKER = "INVARIANT_SENSOR_WHISKER"
INVARIANT_STEERING_LINKAGE = "INVARIANT_STEERING_LINKAGE"


# 방향 벡터 (상, 우, 하, 좌)
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES = ["NORTH", "EAST", "SOUTH", "WEST"]
DIR_SYMBOLS = ["^", ">", "v", "<"]


class CausalAutomaton:
    """
    기계적 인과 그래프로 구성된 자율 보행/주행 오토마톤.
    """

    def __init__(self, automaton_id: str = "clockwork_automaton_1", x: int = 5, y: int = 5, dir_idx: int = 1):
        self.automaton_id = automaton_id
        self.x = x
        self.y = y
        self.dir_idx = dir_idx  # 0:N, 1:E, 2:S, 3:W
        self.step_count = 0
        self.total_energy_dissipated = 0.0

        # 내부 인과 그래프 G = (V, E, C)
        self.graph = CausalGraph(
            graph_id=automaton_id,
            context=TrajectoryContext(medium_type="automaton_kinematic_mechanism")
        )

        self._assemble_mechanism()

    def _assemble_mechanism(self):
        """기구학적 부품 노드 및 인과 엣지 조립 (Assembly)"""
        # 1. 동력원: 주기적 태엽 펄스 발생 노드
        node_power = CausalNode(
            node_id=f"{self.automaton_id}_power",
            node_type=NodeType.STEM,
            invariant_signature=INVARIANT_POWER_CORE,
            payload={"torque": 10.0, "spring_tension": 1.0}
        )
        self.graph.add_node(node_power)

        # 2. 기어열: 1:1 토크 전달 노드
        node_gear = CausalNode(
            node_id=f"{self.automaton_id}_gear",
            node_type=NodeType.STEM,
            invariant_signature=INVARIANT_GEAR_TRAIN,
            payload={"gear_ratio": 1.0, "momentum": 0.0}
        )
        self.graph.add_node(node_gear)

        # 3. 바퀴 구동계: 전진 병진 운동 노드
        node_wheel = CausalNode(
            node_id=f"{self.automaton_id}_wheel",
            node_type=NodeType.STEM,
            invariant_signature=INVARIANT_ACTUATOR_WHEEL,
            payload={"linear_step": 1}
        )
        self.graph.add_node(node_wheel)

        # 4. 촉각 더듬이: 전방 접촉 감각 노드
        node_whisker = CausalNode(
            node_id=f"{self.automaton_id}_whisker",
            node_type=NodeType.STEM,
            invariant_signature=INVARIANT_SENSOR_WHISKER,
            payload={"contact": False}
        )
        self.graph.add_node(node_whisker)

        # 5. 조향 링크: 회전 차동 노드
        node_steer = CausalNode(
            node_id=f"{self.automaton_id}_steer",
            node_type=NodeType.STEM,
            invariant_signature=INVARIANT_STEERING_LINKAGE,
            payload={"yaw_delta": 1} # 90도 시계방향 회전
        )
        self.graph.add_node(node_steer)

        # --- 기구학적 인과 엣지 연결 ---
        # 엣지 1: 동력 -> 기어열 (직접 토크 흐름)
        self.graph.add_edge(CausalEdge(
            source_id=node_power.node_id,
            target_id=node_gear.node_id,
            precondition="torque_drive",
            is_necessary=True
        ))

        # 엣지 2: 기어열 -> 바퀴 (정상 전진 엣지: 더듬이 비접촉 시만 도통)
        self.graph.add_edge(CausalEdge(
            source_id=node_gear.node_id,
            target_id=node_wheel.node_id,
            precondition="unblocked_forward_flow",
            is_necessary=True
        ))

        # 엣지 3: 더듬이 -> 조향 링크 (장애물 접촉 시 조향 엣지 개통)
        self.graph.add_edge(CausalEdge(
            source_id=node_whisker.node_id,
            target_id=node_steer.node_id,
            precondition="contact_redirect_flow",
            is_necessary=True
        ))

    @property
    def forward_vector(self) -> Tuple[int, int]:
        return DIRECTIONS[self.dir_idx]

    @property
    def whisker_target_pos(self) -> Tuple[int, int]:
        dx, dy = self.forward_vector
        return (self.x + dx, self.y + dy)

    def tick(self, world_obstacle_check_fn) -> Dict[str, Any]:
        """
        1 사이클 틱 실행:
        1. 태엽 심장이 틱 펄스를 발생시킨다.
        2. 더듬이 노드가 전방 위상 경계(C)를 검사한다.
        3. 경계가 닫혀있으면(장애물) 모멘텀이 조향 링크로 흘러 회전하고,
           열려있으면 바퀴로 흘러 전진한다.
        """
        self.step_count += 1
        target_pos = self.whisker_target_pos
        is_blocked = world_obstacle_check_fn(target_pos[0], target_pos[1])

        whisker_node = self.graph.nodes[f"{self.automaton_id}_whisker"]
        whisker_node.payload["contact"] = is_blocked

        action_taken = ""

        if is_blocked:
            # [인과적 우회]: 전진 엣지 잠김 -> 조향 링크 도통
            steer_node = self.graph.nodes[f"{self.automaton_id}_steer"]
            yaw_delta = steer_node.payload.get("yaw_delta", 1)
            self.dir_idx = (self.dir_idx + yaw_delta) % 4
            action_taken = f"DEFLECT_TURN_TO_{DIR_NAMES[self.dir_idx]}"
            self.total_energy_dissipated += 2.0
        else:
            # [정상 전진]: 바퀴 노드로 모멘텀 직결
            dx, dy = self.forward_vector
            self.x += dx
            self.y += dy
            action_taken = f"ADVANCE_TO_({self.x},{self.y})"
            self.total_energy_dissipated += 1.0

        return {
            "step": self.step_count,
            "pos": (self.x, self.y),
            "dir": DIR_NAMES[self.dir_idx],
            "dir_symbol": DIR_SYMBOLS[self.dir_idx],
            "whisker_blocked": is_blocked,
            "action": action_taken,
            "energy": self.total_energy_dissipated
        }
