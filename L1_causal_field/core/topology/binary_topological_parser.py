"""
Elysia Binary Memory Topological Parser (8-bit / 16-bit Irreducible Medium)
===========================================================================
바이트(Byte) 및 워드(Word) 단위의 메모리 물리 배열을 읽어
1:1 결정론적 위상 사상(Deterministic Causal Mapping)을 수행하는 바이너리 파서.
부동소수점 근사 없이 메모리 오프셋 전이와 값의 불변 상태 궤적 G = (V, E, C)를 구성한다.
"""

from typing import Dict, List, Set, Any, Optional, Union
from core.topology.causal_stem_branch_engine import (
    CausalGraph, CausalNode, CausalEdge, NodeType, TrajectoryContext
)
from core.topology.base_topological_parser import (
    BaseTopologicalParser,
    INVARIANT_MEM_OFFSET,
    INVARIANT_BYTE_VAL,
    INVARIANT_MEM_BLOCK
)


class BinaryTopologicalParser(BaseTopologicalParser):
    """
    8비트/16비트 바이너리 버퍼를 인과 그래프 G = (V, E, C)로 변환하는 파서.
    """

    def __init__(self, unit_bytes: int = 1):
        super().__init__(medium_type="binary_memory")
        self.unit_bytes = unit_bytes

    def parse(self, source: Union[bytes, bytearray, List[int]]) -> CausalGraph:
        return self.parse_bytes(source)

    def parse_bytes(
        self,
        raw_bytes: Union[bytes, bytearray, List[int]],
        base_offset: int = 0
    ) -> CausalGraph:
        self.reset_counter()
        graph = CausalGraph(
            graph_id="binary_causal_graph",
            context=TrajectoryContext(
                medium_type=self.medium_type,
                environmental_constraints={"unit_bytes": self.unit_bytes, "base_offset": hex(base_offset)}
            )
        )

        byte_data = bytes(raw_bytes) if not isinstance(raw_bytes, bytes) else raw_bytes
        last_node_id: Optional[str] = None

        step = self.unit_bytes
        for offset in range(0, len(byte_data), step):
            chunk = byte_data[offset:offset+step]
            node_id = self._generate_node_id("N_BYTE")
            curr_addr = base_offset + offset

            # 불변 서명: 블록 또는 바이트 값
            inv_sig = INVARIANT_BYTE_VAL if step == 1 else INVARIANT_MEM_BLOCK

            node = CausalNode(
                node_id=node_id,
                node_type=NodeType.STEM,
                invariant_signature=inv_sig,
                payload={
                    "offset": curr_addr,
                    "hex_offset": hex(curr_addr),
                    "hex_value": chunk.hex(),
                    "int_value": int.from_bytes(chunk, byteorder="big")
                }
            )
            graph.add_node(node)
            graph.context_constraints[node_id].add(f"mem_offset:{hex(curr_addr)}")
            graph.context_constraints[node_id].add(f"byte_len:{len(chunk)}")

            # 순차적 주소 오프셋 전이 엣지 형성
            if last_node_id:
                graph.add_edge(CausalEdge(
                    source_id=last_node_id,
                    target_id=node_id,
                    precondition=f"offset_step:+{step}",
                    is_necessary=True
                ))
            last_node_id = node_id

        return graph
