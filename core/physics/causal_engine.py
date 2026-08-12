import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Optional, Tuple

@dataclass(frozen=True)
class CausalNode:
    """
    [CausalNode - 상수의 지형]
    시스템의 변하지 않는 절대적 기하학적 주소와 존재론적 규격입니다.
    - id: 노드의 고유 식별 주소
    - capacity: 이 노드가 감당할 수 있는 최대 포텐셜 용량
    - chromatic_base: 이 노드의 기저 색채 시그니처 [Red, Blue, Yellow]
    """
    id: str
    capacity: float = 1.0
    chromatic_base: Tuple[float, float, float] = (0.33, 0.33, 0.34)


@dataclass(frozen=True)
class TransitionRule:
    """
    [TransitionRule - 인과적 통로]
    노드 간에 정보와 에너지가 흐를 수 있는 허용선이자 구속조건(Constraint)입니다.
    - source_id: 출발 노드 ID
    - target_id: 도착 노드 ID
    - max_flow_rate: 단위 시간당 전송 가능한 최대 포텐셜
    - conductance: 통로의 전도도 (기저 흐름 저항의 역수)
    """
    source_id: str
    target_id: str
    max_flow_rate: float = 0.5
    conductance: float = 1.0


@dataclass
class CausalState:
    """
    [CausalState - 흐르는 대지]
    시공간 위에서 끊임없이 출렁이는 가변적인 시스템의 전체 상태(State)입니다.
    - potentials: 각 노드 ID별 현재 잠재 에너지(Potential)
    - velocities: 각 노드 ID별 에너지 이동 속도(Momentum)
    - chromatics: 각 노드 ID별 동적 색채 시그니처 (Red/Blue/Yellow)
    """
    potentials: Dict[str, float] = field(default_factory=dict)
    velocities: Dict[str, Dict[str, float]] = field(default_factory=dict) # Node -> Neighbor -> flow velocity
    chromatics: Dict[str, np.ndarray] = field(default_factory=dict)

    def copy(self) -> 'CausalState':
        return CausalState(
            potentials=self.potentials.copy(),
            velocities={nid: flows.copy() for nid, flows in self.velocities.items()},
            chromatics={nid: np.copy(chroma) for nid, chroma in self.chromatics.items()}
        )


@dataclass(frozen=True)
class StateDelta:
    """
    [StateDelta - 국소적 마찰량]
    찰나의 에이전트가 우주적 대지에 가하고 사라지는 가변 저항의 델타(Delta)입니다.
    - potential_diffs: 각 노드에 가해진 잠재 에너지 변화량
    - velocity_diffs: 노드 간 채널 흐름의 가속도/속도 변화량
    - chromatic_diffs: 색채 시그니처의 변화량
    """
    potential_diffs: Dict[str, float] = field(default_factory=dict)
    velocity_diffs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    chromatic_diffs: Dict[str, np.ndarray] = field(default_factory=dict)


# 원자적 함수(Atomic Action): f(State) -> StateDelta
# 찰나의 마찰을 처리하고 즉시 소멸하는 단절된 에이전트용 순수 작업 단위의 규격입니다.
AtomicAction = Callable[[CausalState], StateDelta]


class CausalEngine:
    """
    [CausalEngine - 전체화 엔진]
    단절된 에이전트들이 일으킨 찰나의 델타들을 인과적 구속조건(TransitionRule) 하에 검증 및 정류하고,
    시스템 스스로 최소 작용의 원리(Principle of Least Action)를 따르며 연속적인 평형을 찾아가도록
    상태를 영속적으로 통합·회전시키는 거대한 기어 장치입니다.
    """
    def __init__(self, nodes: List[CausalNode], rules: List[TransitionRule]):
        self.nodes = {node.id: node for node in nodes}
        self.rules: Dict[Tuple[str, str], TransitionRule] = {}
        self.adjacency: Dict[str, List[str]] = {node.id: [] for node in nodes}

        # 규칙 및 인접 노드 구조 구축
        for rule in rules:
            if rule.source_id in self.nodes and rule.target_id in self.nodes:
                self.rules[(rule.source_id, rule.target_id)] = rule
                self.adjacency[rule.source_id].append(rule.target_id)
                # 양방향 흐름이 기본이므로, 역방향 규칙도 존재하지 않는다면 동일 Conductance로 가상 등록
                if (rule.target_id, rule.source_id) not in self.rules:
                    self.adjacency[rule.target_id].append(rule.source_id)

        # 초기 상태 선포
        self.state = CausalState()
        for node in nodes:
            self.state.potentials[node.id] = 0.0
            self.state.chromatics[node.id] = np.array(node.chromatic_base, dtype=np.float32)
            self.state.velocities[node.id] = {neighbor: 0.0 for neighbor in self.adjacency[node.id]}

    def apply_action(self, action: AtomicAction) -> StateDelta:
        """
        단절된 에이전트(Action)를 호출하여 찰나의 델타를 획득하고,
        이를 전체화 엔진의 인과 법칙(구속조건)에 맞게 정류하여 영속 상태에 투사합니다.
        """
        # 1. 에이전트는 오직 읽기 전용 상태만을 관측함
        read_only_state = self.state.copy()

        # 2. 찰나의 마찰(Delta) 발생 후 에이전트는 즉시 메모리에서 소멸
        raw_delta = action(read_only_state)

        # 3. 인과 장벽(Transition Constraint) 검증 및 정류 (Rectification)
        rectified_delta = self._rectify_delta(raw_delta)

        # 4. 정류된 델타를 영속적 대지(State) 위에 통합
        self._integrate_delta(rectified_delta)

        return rectified_delta

    def step(self, dt: float = 0.1):
        """
        [최소 작용의 원리 및 포텐셜 유동]
        외부 자극이 없는 상황에서도, 시스템은 이미 형성된 인과적 기울기(Potential Gradient)와
        전도도(Conductance)를 따라 스스로 에너지를 최소화하는 연속적 물리 이완(Relaxation) 단계를 밟습니다.
        """
        next_potentials = self.state.potentials.copy()
        next_chromatics = {nid: np.copy(chroma) for nid, chroma in self.state.chromatics.items()}
        processed_edges = set()

        for node_id, node in self.nodes.items():
            curr_pot = self.state.potentials[node_id]
            neighbors = self.adjacency[node_id]

            if not neighbors:
                continue

            # 포텐셜 구배(Gradient)에 따른 유동 및 가속도 계산
            for neighbor_id in neighbors:
                # 중복 전이 방지 (A->B 와 B->A 가 한 루프 내에서 중복 실행되는 물리적 왜곡을 제어)
                edge_key = tuple(sorted((node_id, neighbor_id)))
                if edge_key in processed_edges:
                    continue
                processed_edges.add(edge_key)

                neighbor_pot = self.state.potentials[neighbor_id]
                pot_diff = curr_pot - neighbor_pot

                # 연결 통로의 구속 규칙 조회
                rule = self.rules.get((node_id, neighbor_id))
                if not rule:
                    # 역방향 가상 규칙
                    rule = self.rules.get((neighbor_id, node_id))
                    conductance = rule.conductance if rule else 1.0
                    max_flow = rule.max_flow_rate if rule else 0.5
                else:
                    conductance = rule.conductance
                    max_flow = rule.max_flow_rate

                # 유동 속도(Flow Velocity) 업데이트: F = -k * grad(V)
                # 이전 속도의 관성(Inertia)을 80% 유지하고, 신규 포텐셜 구배 가속도를 20% 더함
                curr_flow_vel = self.state.velocities[node_id].get(neighbor_id, 0.0)
                target_flow_vel = pot_diff * conductance

                new_flow_vel = (curr_flow_vel * 0.8) + (target_flow_vel * 0.2)

                # 최대 흐름 한계 구속조건 적용 (Capacity constraint)
                new_flow_vel = np.clip(new_flow_vel, -max_flow, max_flow)

                # 대칭 흐름 보존
                self.state.velocities[node_id][neighbor_id] = float(new_flow_vel)
                if neighbor_id in self.state.velocities and node_id in self.state.velocities[neighbor_id]:
                    self.state.velocities[neighbor_id][node_id] = float(-new_flow_vel)

                # 포텐셜 전이량 계산: Q = v * dt
                flow_transfer = new_flow_vel * dt

                # 노드의 한계 용량(Capacity)을 초과하지 않도록 보정
                next_potentials[node_id] = float(np.clip(next_potentials[node_id] - flow_transfer, 0.0, node.capacity))
                next_potentials[neighbor_id] = float(np.clip(next_potentials[neighbor_id] + flow_transfer, 0.0, self.nodes[neighbor_id].capacity))

                # 색채 시그니처의 전이 (Flux/Order의 동반 유출입)
                if flow_transfer > 0:
                    # 노드 A에서 B로 유출: A의 Flux(Red)가 전이되며 에너지가 흘러감
                    chroma_diff = self.state.chromatics[node_id] * (flow_transfer * 0.1)
                    next_chromatics[node_id] = np.clip(next_chromatics[node_id] - chroma_diff, 0.0, 1.0)
                    next_chromatics[neighbor_id] = np.clip(next_chromatics[neighbor_id] + chroma_diff, 0.0, 1.0)
                elif flow_transfer < 0:
                    # 노드 B에서 A로 유입
                    amt = abs(flow_transfer)
                    chroma_diff = self.state.chromatics[neighbor_id] * (amt * 0.1)
                    next_chromatics[neighbor_id] = np.clip(next_chromatics[neighbor_id] - chroma_diff, 0.0, 1.0)
                    next_chromatics[node_id] = np.clip(next_chromatics[node_id] + chroma_diff, 0.0, 1.0)

        # 상태 업데이트 및 정규화
        self.state.potentials = next_potentials
        for nid in self.nodes:
            tot = np.sum(next_chromatics[nid])
            if tot > 0:
                self.state.chromatics[nid] = next_chromatics[nid] / tot
            else:
                self.state.chromatics[nid] = np.array(self.nodes[nid].chromatic_base, dtype=np.float32)

    def _rectify_delta(self, raw_delta: StateDelta) -> StateDelta:
        """
        에이전트의 델타 제안이 시스템의 물리적 임계나 구속조건을 위반하는지 검증하고 정류합니다.
        """
        rectified_pot_diffs = {}
        rectified_vel_diffs = {}
        rectified_chrom_diffs = {}

        # 1. 포텐셜 델타 정류 (노드의 최대 용량 및 물리적 하한선 검증)
        for nid, diff in raw_delta.potential_diffs.items():
            if nid not in self.nodes:
                continue
            curr_val = self.state.potentials[nid]
            node_capacity = self.nodes[nid].capacity

            # 다음 포텐셜 예측값 계산 및 클램핑
            projected = curr_val + diff
            clamped = np.clip(projected, 0.0, node_capacity)
            rectified_pot_diffs[nid] = float(clamped - curr_val)

        # 2. 채널 유동 속도 델타 정류 (채널 존재성 및 속도 제약조건 검증)
        for src_id, flow_diffs in raw_delta.velocity_diffs.items():
            if src_id not in self.nodes:
                continue
            rectified_vel_diffs[src_id] = {}
            for target_id, vel_diff in flow_diffs.items():
                if target_id not in self.adjacency[src_id]:
                    continue  # 연결되지 않은 노드 간의 유동은 원천 차단

                rule = self.rules.get((src_id, target_id)) or self.rules.get((target_id, src_id))
                max_flow = rule.max_flow_rate if rule else 0.5

                curr_vel = self.state.velocities[src_id].get(target_id, 0.0)
                projected_vel = curr_vel + vel_diff
                clamped_vel = np.clip(projected_vel, -max_flow, max_flow)

                rectified_vel_diffs[src_id][target_id] = float(clamped_vel - curr_vel)

        # 3. 색채 델타 정류
        for nid, chroma_diff in raw_delta.chromatic_diffs.items():
            if nid not in self.nodes:
                continue
            # 색채 합이 0을 유지하도록 혹은 기하학적 형태에 맞게 단순 클램핑 후 보관
            rectified_chrom_diffs[nid] = np.clip(chroma_diff, -0.5, 0.5)

        return StateDelta(
            potential_diffs=rectified_pot_diffs,
            velocity_diffs=rectified_vel_diffs,
            chromatic_diffs=rectified_chrom_diffs
        )

    def _integrate_delta(self, delta: StateDelta):
        """
        정류된 가변 델타들을 기저 상태에 반영하여 하나의 연속적인 역사적 선으로 합성합니다.
        """
        for nid, diff in delta.potential_diffs.items():
            self.state.potentials[nid] = float(np.clip(self.state.potentials[nid] + diff, 0.0, self.nodes[nid].capacity))

        for src_id, flow_diffs in delta.velocity_diffs.items():
            for target_id, diff in flow_diffs.items():
                curr_vel = self.state.velocities[src_id].get(target_id, 0.0)
                rule = self.rules.get((src_id, target_id)) or self.rules.get((target_id, src_id))
                max_flow = rule.max_flow_rate if rule else 0.5
                self.state.velocities[src_id][target_id] = float(np.clip(curr_vel + diff, -max_flow, max_flow))

        for nid, chroma_diff in delta.chromatic_diffs.items():
            new_chroma = self.state.chromatics[nid] + chroma_diff
            tot = np.sum(new_chroma)
            if tot > 0:
                self.state.chromatics[nid] = np.clip(new_chroma, 0.0, 1.0) / tot
            else:
                self.state.chromatics[nid] = np.array(self.nodes[nid].chromatic_base, dtype=np.float32)
