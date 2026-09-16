"""
Elysia Hierarchical Binary Lattice Engine
=========================================
8비트/16비트의 최소 불변 단위(Irreducible Causal Unit)에서 출발하여
4KB 메모리 페이지, 64KB/1MB 대규모 데이터 청크, 나아가 기가바이트/테라바이트 수준의
초대규모 파일 구조와 게임 월드를 메모리 고갈 없이 계층적으로 다루는 인과 격자 엔진.
"""

from typing import Dict, List, Set, Any, Optional, Tuple, Union
from collections import defaultdict
import math

from core.topology.causal_stem_branch_engine import (
    CausalGraph, CausalNode, CausalEdge, NodeType, TrajectoryContext
)
from core.topology.base_topological_parser import (
    INVARIANT_MEM_OFFSET,
    INVARIANT_BYTE_VAL,
    INVARIANT_MEM_BLOCK
)


class HierarchicalLevel:
    UNIT = 0      # 1 Byte / 16-bit Word
    BLOCK = 1     # 64B ~ 4KB Voxel / Page
    CHUNK = 2     # 64KB ~ 1MB Macro Section / World Sector
    LATTICE = 3   # Whole File / World Envelope


class CompoundLatticeNode(CausalNode):
    """
    하위 인과 격자들을 캡슐화한 복합 인과 노드 (Hierarchical Compound Node).
    실제 하위 바이트를 지연 전개(Lazy Materialization)하여 테라바이트급 데이터도
    최소한의 메모리로 관리한다.
    """

    def __init__(
        self,
        node_id: str,
        level: int,
        offset_range: Tuple[int, int],
        summary_signature: str = INVARIANT_MEM_BLOCK,
        parent_id: Optional[str] = None
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.STEM,
            invariant_signature=summary_signature,
            payload={
                "level": level,
                "start_offset": offset_range[0],
                "end_offset": offset_range[1],
                "size_bytes": offset_range[1] - offset_range[0],
                "dirty": True
            }
        )
        self.level = level
        self.offset_range = offset_range
        self.parent_id = parent_id
        self.child_node_ids: List[str] = []
        self.is_expanded: bool = False

    @property
    def is_dirty(self) -> bool:
        return self.payload.get("dirty", False)

    def mark_dirty(self, dirty: bool = True):
        self.payload["dirty"] = dirty


class HierarchicalBinaryLattice:
    """
    프랙탈 계층형 인과 격자 (Hierarchical Binary Lattice).
    실제 지연 전개(Lazy Materialization)를 적용하여 초기 할당 시 노드를 생성하지 않고
    오프셋 접근 시 필요 노드만 온디맨드로 생성합니다.
    """

    def __init__(self, total_size_bytes: int, block_size: int = 4096, chunk_size: int = 65536):
        self.total_size = total_size_bytes
        self.block_size = block_size
        self.chunk_size = chunk_size
        self.graph = CausalGraph(
            graph_id="hierarchical_binary_lattice",
            context=TrajectoryContext(
                medium_type="hierarchical_memory",
                environmental_constraints={
                    "total_size": total_size_bytes,
                    "block_size": block_size,
                    "chunk_size": chunk_size
                }
            )
        )
        self.compound_nodes: Dict[str, CompoundLatticeNode] = {}
        self.dirty_nodes: Set[str] = set()

    def get_or_create_chunk_node(self, chunk_idx: int) -> CompoundLatticeNode:
        """지연 전개(Lazy Materialization): 청크 요청 시 노드 생성"""
        chunk_id = f"CHUNK_{chunk_idx}"
        if chunk_id in self.compound_nodes:
            return self.compound_nodes[chunk_id]

        c_start = chunk_idx * self.chunk_size
        c_end = min(self.total_size, (chunk_idx + 1) * self.chunk_size)

        chunk_node = CompoundLatticeNode(
            node_id=chunk_id,
            level=HierarchicalLevel.CHUNK,
            offset_range=(c_start, c_end),
            summary_signature="INVARIANT_MEM_CHUNK"
        )
        self.compound_nodes[chunk_id] = chunk_node
        self.graph.add_node(chunk_node)
        return chunk_node

    def get_or_create_block_node(self, chunk_idx: int, block_idx: int) -> CompoundLatticeNode:
        """지연 전개(Lazy Materialization): 블록 요청 시 노드 생성"""
        block_id = f"BLOCK_{chunk_idx}_{block_idx}"
        if block_id in self.compound_nodes:
            return self.compound_nodes[block_id]

        chunk_node = self.get_or_create_chunk_node(chunk_idx)
        c_start, c_end = chunk_node.offset_range

        b_start = c_start + (block_idx * self.block_size)
        b_end = min(c_end, b_start + self.block_size)

        block_node = CompoundLatticeNode(
            node_id=block_id,
            level=HierarchicalLevel.BLOCK,
            offset_range=(b_start, b_end),
            summary_signature=INVARIANT_MEM_BLOCK,
            parent_id=chunk_node.node_id
        )
        self.compound_nodes[block_id] = block_node
        self.graph.add_node(block_node)
        chunk_node.child_node_ids.append(block_id)
        return block_node

    def perturb_offset(self, offset: int, new_value_payload: Any) -> List[str]:
        """
        지연 전개 구조로 특정 오프셋에 변위 발생 시 필요 청크 및 블록 노드만 동적 실체화
        """
        if offset < 0 or offset >= self.total_size:
            raise ValueError(f"Offset {offset} is out of bounds [0, {self.total_size})")

        c_idx = offset // self.chunk_size
        b_idx = (offset % self.chunk_size) // self.block_size

        chunk_node = self.get_or_create_chunk_node(c_idx)
        block_node = self.get_or_create_block_node(c_idx, b_idx)

        chunk_node.mark_dirty(True)
        self.dirty_nodes.add(chunk_node.node_id)

        block_node.mark_dirty(True)
        block_node.payload["last_value"] = new_value_payload
        self.dirty_nodes.add(block_node.node_id)

        return [chunk_node.node_id, block_node.node_id]

    def consume_dirty_deltas(self) -> List[CompoundLatticeNode]:
        """변경된 인과 노드들만 회수하고 Dirty 상태 소진"""
        deltas = [self.compound_nodes[nid] for nid in self.dirty_nodes if nid in self.compound_nodes]
        for node in deltas:
            node.mark_dirty(False)
        self.dirty_nodes.clear()
        return deltas
