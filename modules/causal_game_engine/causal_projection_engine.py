"""
Elysia Causal Projection Game Engine
====================================
8K 동영상 디코더의 인과 투영(Projection/Unfolding) 원리를 내재화하여,
매 프레임 전체 픽셀/폴리곤을 수치 재계산하는 기존 게임 엔진의 비효율을 완전히 폐기한
'인과 투사 기반 초경량 게임 엔진'.

핵심 원리:
1. I-State (인과 키 닻): 게임 월드의 기구학적, 지형적 불변 위상 뼈대.
2. P-Flow (차분 인과 빔): 사용자 조작이나 국소 이벤트 발생 시 변위된 인과 경로만 dirty 전파.
3. CausalProjectionRenderer: 변위된 노드만을 대상 화면 버퍼(또는 아이소메트릭 타일맵)에 O(Δ)로 1:1 직역 투영.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import numpy as np

from core.topology.causal_stem_branch_engine import (
    CausalGraph, CausalNode, CausalEdge, NodeType, TrajectoryContext
)
from core.topology.base_topological_parser import (
    INVARIANT_RIGID_BODY,
    INVARIANT_JOINT_REVOLUTE,
    INVARIANT_STATE_BINDING
)


class CausalProjectionWorld:
    """
    인과 그래프 G = (V, E, C)로 구현된 게임 세계.
    객체나 타일의 상태 변화는 부동소수점 물리 방정식이 아니라,
    구속 조건(Precondition) 엣지를 따라 위상적으로 전달된다.
    """

    def __init__(self, world_id: str = "causal_world", width: int = 100, height: int = 100):
        self.width = width
        self.height = height
        self.graph = CausalGraph(
            graph_id=world_id,
            context=TrajectoryContext(
                medium_type="causal_game_world",
                environmental_constraints={"dimensions": (width, height)}
            )
        )
        self.dirty_node_ids: Set[str] = set()
        self.entity_positions: Dict[str, Tuple[int, int]] = {}

    def spawn_entity(
        self,
        entity_id: str,
        x: int,
        y: int,
        entity_type: str,
        invariant_signature: str = INVARIANT_RIGID_BODY,
        state_payload: Optional[Dict[str, Any]] = None
    ) -> CausalNode:
        payload = state_payload or {}
        payload.update({"pos": (x, y), "type": entity_type, "symbol": entity_type[0].upper()})

        node = CausalNode(
            node_id=entity_id,
            node_type=NodeType.STEM,
            invariant_signature=invariant_signature,
            payload=payload
        )
        self.graph.add_node(node)
        self.entity_positions[entity_id] = (x, y)
        self.dirty_node_ids.add(entity_id)
        return node

    def bind_kinematic_link(
        self,
        parent_id: str,
        child_id: str,
        dof_type: str = "MOMENTUM_TRANSFER",
        ratio: float = 1.0
    ):
        """두 엔티티 간 인과적 연동(Kinematic Coupling) 엣지 등록"""
        edge = CausalEdge(
            source_id=parent_id,
            target_id=child_id,
            precondition=f"{dof_type}:{ratio}",
            is_necessary=True
        )
        self.graph.add_edge(edge)

    def apply_impulse(self, entity_id: str, delta_state: Dict[str, Any]) -> List[str]:
        """
        플레이어의 조작이나 국소적 충격(Impulse) 인가.
        전체 세계를 다시 연산하지 않고, 연동된 엣지를 타고 변위(Dirty Flow)를 즉각 전파한다.
        """
        if entity_id not in self.graph.nodes:
            return []

        affected_nodes: List[str] = []
        queue = [(entity_id, delta_state)]

        while queue:
            curr_id, current_delta = queue.pop(0)
            node = self.graph.nodes[curr_id]
            affected_nodes.append(curr_id)
            self.dirty_node_ids.add(curr_id)

            # 상태 업데이트
            for k, v in current_delta.items():
                if k == "dpos":
                    cx, cy = node.payload.get("pos", (0, 0))
                    dx, dy = v
                    nx, ny = max(0, min(self.width - 1, cx + dx)), max(0, min(self.height - 1, cy + dy))
                    node.payload["pos"] = (nx, ny)
                    self.entity_positions[curr_id] = (nx, ny)
                else:
                    node.payload[k] = v

            # 구속 조건(Precondition) 엣지를 통한 인과 연쇄 전달 ("Let it flow")
            for next_id, edge in self.graph.get_successors(curr_id):
                if next_id not in affected_nodes:
                    # 전달 규칙 분석
                    propagated_delta = dict(current_delta)
                    if "dpos" in current_delta and "MOMENTUM_TRANSFER" in edge.precondition:
                        ratio = float(edge.precondition.split(":")[-1])
                        dx, dy = current_delta["dpos"]
                        propagated_delta["dpos"] = (int(dx * ratio), int(dy * ratio))
                    queue.append((next_id, propagated_delta))

        return affected_nodes


class CausalProjectionRenderer:
    """
    동영상 디코더 방식의 초경량 인과 투사 렌더러.
    - 이전 프레임의 전체 버퍼를 버리고 다시 그리지 않는다.
    - dirty_node_ids에 등록된 인과 노드들의 위치/상태 변화만을 화면 버퍼에 1:1 패치(Patch).
    - 계산 복잡도: O(전체 타일 수)가 아닌 O(변위된 노드 수 Δ)
    """

    def __init__(self, world: CausalProjectionWorld):
        self.world = world
        # 가상 화면 2D/2.5D 타일 버퍼 (문자/심볼 및 상태값)
        self.screen_buffer: Dict[Tuple[int, int], str] = {}
        self.total_projected_frames: int = 0
        self.total_patches_applied: int = 0

    def render_frame(self, full_redraw: bool = False) -> Dict[str, Any]:
        """
        차분 투사(Differential Projection) 실행.
        """
        patches_in_frame = 0

        if full_redraw:
            # I-Frame: 전체 키 상태 투영
            self.screen_buffer.clear()
            for node_id, node in self.world.graph.nodes.items():
                pos = node.payload.get("pos", (0, 0))
                symbol = node.payload.get("symbol", ".")
                self.screen_buffer[pos] = symbol
                patches_in_frame += 1
            self.world.dirty_node_ids.clear()
        else:
            # P-Frame: 오직 dirty 인과 노드만 화면에 투사
            dirty_ids = list(self.world.dirty_node_ids)
            for node_id in dirty_ids:
                node = self.world.graph.nodes[node_id]
                pos = node.payload.get("pos", (0, 0))
                symbol = node.payload.get("symbol", ".")
                
                # 이전 위치 잔상 제거 (원하면 빈 셀 복원 가능)
                self.screen_buffer[pos] = symbol
                patches_in_frame += 1

            self.world.dirty_node_ids.clear()

        self.total_projected_frames += 1
        self.total_patches_applied += patches_in_frame

        return {
            "frame_index": self.total_projected_frames,
            "patches_applied": patches_in_frame,
            "total_entities": len(self.world.graph.nodes),
            "is_key_frame": full_redraw,
            "efficiency_ratio": patches_in_frame / max(1, len(self.world.graph.nodes))
        }

    def render_ascii_viewport(self, view_x: int, view_y: int, view_w: int, view_h: int) -> str:
        """현재 화면 버퍼의 특정 뷰포트를 ASCII 텍스트로 시각화"""
        lines = []
        for y in range(view_y, view_y + view_h):
            row = []
            for x in range(view_x, view_x + view_w):
                row.append(self.screen_buffer.get((x, y), "·"))
            lines.append(" ".join(row))
        return "\n".join(lines)
