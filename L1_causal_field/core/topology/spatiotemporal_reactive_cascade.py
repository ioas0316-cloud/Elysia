"""
Elysia Core Architecture: SpatioTemporal Sync Adapter & Reactive Cascade

This module implements the SpatioTemporal Sync Adapter and Reactive Cascade DAG,
enabling non-symbolic / multi-modal frame alignment and Kahn's topological sort
based reactive propagation from root nodes through causal integrity to action decisions.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set, Tuple


# ============================================================================
# 1. State and Frame Data Structures
# ============================================================================

@dataclass
class SemanticState:
    identity: str
    qualities: Set[str] = field(default_factory=set)
    relational_bindings: Dict[str, str] = field(default_factory=dict)


@dataclass
class VisualFrame:
    timestamp_ms: float
    bbox: Tuple[float, float, float, float]
    label: str


@dataclass
class AudioFrame:
    timestamp_ms: float
    doa_angle_deg: float
    decibel: float
    sound_type: str


# ============================================================================
# 2. SpatioTemporal Sync Adapter
# ============================================================================

class SpatioTemporalSyncAdapter:
    def __init__(
        self,
        max_time_delta_ms: float = 150.0,
        camera_fov_deg: float = 90.0,
        max_angle_delta_deg: float = 15.0,
    ):
        self.max_time_delta_ms = max_time_delta_ms
        self.camera_fov_deg = camera_fov_deg
        self.max_angle_delta_deg = max_angle_delta_deg

    def _calculate_visual_angle(self, bbox: Tuple[float, float, float, float]) -> float:
        x_center = (bbox[0] + bbox[2]) / 2.0
        return 180.0 + (x_center - 0.5) * self.camera_fov_deg

    def process_sync(self, v_frame: VisualFrame, a_frame: AudioFrame) -> Dict[str, str]:
        time_delta = abs(v_frame.timestamp_ms - a_frame.timestamp_ms)
        is_temporal_synced = time_delta <= self.max_time_delta_ms

        v_angle = self._calculate_visual_angle(v_frame.bbox)
        angle_delta = abs(v_angle - a_frame.doa_angle_deg)
        is_spatial_synced = angle_delta <= self.max_angle_delta_deg

        bindings = {
            "TEMPORAL_SYNC": "PASS" if is_temporal_synced else "FAIL",
            "SPATIAL_SYNC": "PASS" if is_spatial_synced else "FAIL",
            "TEMPORAL_DELTA_MS": f"{time_delta:.1f}",
            "SPATIAL_ANGLE_DELTA": f"{angle_delta:.1f}deg",
        }

        if is_temporal_synced and is_spatial_synced:
            bindings["CAUSAL_INTEGRITY"] = "COHERENT_SINGLE_SOURCE"
            bindings["SYNC_STATUS"] = "VERIFIED"
        elif is_temporal_synced and not is_spatial_synced:
            bindings["CAUSAL_INTEGRITY"] = "OFF_SCREEN_OR_DUBBED"
            bindings["SYNC_STATUS"] = "SPATIAL_ANOMALY"
        else:
            bindings["CAUSAL_INTEGRITY"] = "DISJOINT_NOISE"
            bindings["SYNC_STATUS"] = "UNCORRELATED"

        return bindings


# ============================================================================
# 3. Reactive DAG Node & Propagation Engine
# ============================================================================

class DAGNode:
    def __init__(self, node_id: str, operator_fn: Callable[[List[SemanticState]], SemanticState]):
        self.node_id = node_id
        self.operator_fn = operator_fn
        self.state: SemanticState = SemanticState(identity=node_id)


class ReactiveDAG:
    def __init__(self):
        self.nodes: Dict[str, DAGNode] = {}
        self.graph: Dict[str, List[str]] = defaultdict(list)
        self.in_degree: Dict[str, int] = defaultdict(int)

    def add_node(self, node: DAGNode):
        self.nodes[node.node_id] = node
        if node.node_id not in self.in_degree:
            self.in_degree[node.node_id] = 0

    def add_edge(self, parent_id: str, child_id: str):
        self.graph[parent_id].append(child_id)
        self.in_degree[child_id] += 1

    def _topological_sort(self) -> List[str]:
        in_deg = self.in_degree.copy()
        queue = deque([n_id for n_id, deg in in_deg.items() if deg == 0])
        sorted_nodes = []

        while queue:
            curr = queue.popleft()
            sorted_nodes.append(curr)
            for neighbor in self.graph[curr]:
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)

        return sorted_nodes

    def propagate(self, root_node_id: str, root_state: SemanticState):
        """입력 노드의 상태 변경에 따른 인과망 위상 연쇄 재계산 전파"""
        self.nodes[root_node_id].state = root_state
        print(f"⚡ [Reactive Cascade Start] Root Node [{root_node_id}] 상태 변동 감지")

        exec_order = self._topological_sort()
        start_idx = exec_order.index(root_node_id) if root_node_id in exec_order else 0

        for node_id in exec_order[start_idx:]:
            if node_id == root_node_id:
                continue

            # 상위 의존 노드들의 연산 결과 수집
            parent_states = []
            for p_id, children in self.graph.items():
                if node_id in children:
                    parent_states.append(self.nodes[p_id].state)

            # 연산자 전파 실행 및 상태 연쇄 갱신
            new_state = self.nodes[node_id].operator_fn(parent_states)
            self.nodes[node_id].state = new_state
            print(f"  ├─ 🔄 [{node_id}] 연산 재계산 완료")
            print(f"  │    ├─ Qualities: {new_state.qualities}")
            print(f"  │    └─ Bindings : {new_state.relational_bindings}")


# ============================================================================
# 4. Linguistic Operators
# ============================================================================

def sync_input_operator(
    adapter: SpatioTemporalSyncAdapter,
    v_frame: VisualFrame,
    a_frame: AudioFrame,
) -> SemanticState:
    """[입력 연산자] SyncAdapter 출력을 SemanticState 주입"""
    sync_bindings = adapter.process_sync(v_frame, a_frame)
    qualities = {"MULTIMODAL_INPUT"}
    if sync_bindings.get("SYNC_STATUS") == "VERIFIED":
        qualities.add("SYNCED_EVENT")
    else:
        qualities.add("DESYNC_ANOMALY")

    return SemanticState("Sync_Input_Node", qualities, sync_bindings)


def causal_integrity_operator(parent_states: List[SemanticState]) -> SemanticState:
    """[1차 의존 연산자] 시공간 정합성 기반 인과성 신뢰도 평가"""
    parent = parent_states[0]
    integrity = parent.relational_bindings.get("CAUSAL_INTEGRITY")

    qualities = set(parent.qualities)
    bindings = dict(parent.relational_bindings)

    if integrity == "COHERENT_SINGLE_SOURCE":
        qualities.add("HIGH_CAUSAL_CONFIDENCE")
        bindings["AGENT_ATTENTION"] = "FOCUS_TARGET"
    elif integrity == "OFF_SCREEN_OR_DUBBED":
        qualities.add("OUT_OF_FRAME_ATTENTION")
        bindings["AGENT_ATTENTION"] = "SCAN_SURROUNDINGS"
    else:
        qualities.add("NOISE_REJECTION")
        bindings["AGENT_ATTENTION"] = "IGNORE"

    return SemanticState("Causal_Integrity_Node", qualities, bindings)


def action_decision_operator(parent_states: List[SemanticState]) -> SemanticState:
    """[2차 의존 연산자] 최종 자율 주체 행동 결정"""
    parent = parent_states[0]
    attention = parent.relational_bindings.get("AGENT_ATTENTION")

    qualities = set()
    bindings = {}

    if attention == "FOCUS_TARGET":
        qualities.add("ACTION_TRACKING")
        bindings["EXECUTE_COMMAND"] = "LOCK_CAMERA_AND_LISTEN"
    elif attention == "SCAN_SURROUNDINGS":
        qualities.add("ACTION_SEARCH")
        bindings["EXECUTE_COMMAND"] = "PAN_CAMERA_TO_DOA"
    else:
        qualities.add("ACTION_IDLE")
        bindings["EXECUTE_COMMAND"] = "MAINTAIN_CURRENT_STATE"

    return SemanticState("Action_Decision_Node", qualities, bindings)
