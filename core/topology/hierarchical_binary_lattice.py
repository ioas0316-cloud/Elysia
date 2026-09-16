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
    - Level 0: 8-bit/16-bit 원자 단위
    - Level 1: 4KB 페이지/블록
    - Level 2: 1MB 섹터/청크
    - Level 3: 전체 월드/바이너리 엔벨로프
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

        self._build_top_levels()

    def _build_top_levels(self):
        """청크(Level 2) 및 블록(Level 1)의 계층적 뼈대(Skeleton)를 제로-카피로 사전 구축"""
        chunk_count = max(1, math.ceil(self.total_size / self.chunk_size))

        for c_idx in range(chunk_count):
            c_start = c_idx * self.chunk_size
            c_end = min(self.total_size, (c_idx + 1) * self.chunk_size)
            chunk_id = f"CHUNK_{c_idx}"

            chunk_node = CompoundLatticeNode(
                node_id=chunk_id,
                level=HierarchicalLevel.CHUNK,
                offset_range=(c_start, c_end),
                summary_signature="INVARIANT_MEM_CHUNK"
            )
            self.compound_nodes[chunk_id] = chunk_node
            self.graph.add_node(chunk_node)

            # 청크 내 블록(Level 1) 구성
            block_count = math.ceil((c_end - c_start) / self.block_size)
            last_block_id: Optional[str] = None

            for b_idx in range(block_count):
                b_start = c_start + (b_idx * self.block_size)
                b_end = min(c_end, b_start + self.block_size)
                block_id = f"BLOCK_{c_idx}_{b_idx}"

                block_node = CompoundLatticeNode(
                    node_id=block_id,
                    level=HierarchicalLevel.BLOCK,
                    offset_range=(b_start, b_end),
                    summary_signature=INVARIANT_MEM_BLOCK,
                    parent_id=chunk_id
                )
                self.compound_nodes[block_id] = block_node
                self.graph.add_node(block_node)
                chunk_node.child_node_ids.append(block_id)

                # 블록 간 연속성 엣지
                if last_block_id:
                    self.graph.add_edge(CausalEdge(
                        source_id=last_block_id,
                        target_id=block_id,
                        precondition=f"page_stride:+{self.block_size}",
                        is_necessary=True
                    ))
                last_block_id = block_id

    def perturb_offset(self, offset: int, new_value_payload: Any) -> List[str]:
        """
        특정 메모리 오프셋에 입력/변위(Perturbation)가 일어났을 때,
        전체 세계를 탐색하지 않고 해당 계층 경로(Chunk -> Block -> Unit)만 핀포인트로 Dirty 마킹.
        """
        if offset < 0 or offset >= self.total_size:
            raise ValueError(f"Offset {offset} is out of bounds [0, {self.total_size})")

        c_idx = offset // self.chunk_size
        b_idx = (offset % self.chunk_size) // self.block_size

        chunk_id = f"CHUNK_{c_idx}"
        block_id = f"BLOCK_{c_idx}_{b_idx}"

        affected_path = [chunk_id, block_id]

        # Dirty Flag 활성화
        if chunk_id in self.compound_nodes:
            self.compound_nodes[chunk_id].mark_dirty(True)
            self.dirty_nodes.add(chunk_id)

        if block_id in self.compound_nodes:
            self.compound_nodes[block_id].mark_dirty(True)
            self.compound_nodes[block_id].payload["last_value"] = new_value_payload
            self.dirty_nodes.add(block_id)

        return affected_path

    def consume_dirty_deltas(self) -> List[CompoundLatticeNode]:
        """변경된 인과 노드들만 회수하고 Dirty 상태를 소진 (동영상 P-Frame 디코딩 준비)"""
        deltas = [self.compound_nodes[nid] for nid in self.dirty_nodes if nid in self.compound_nodes]
        for node in deltas:
            node.mark_dirty(False)
        self.dirty_nodes.clear()
        return deltas
