"""
Elysia Binary Container Stripper Module
=======================================
복잡한 바이너리 파일 포맷(매직 바이트, 청크 헤더, 섹션 메타데이터, 패리티 체크섬)을
순수하게 분리하여 맥락 제약(C: Context Constraints)으로 격리하고,
내재된 순수 불변 데이터 궤적(Stem: V, E)만을 적출하는 바이너리 박리 엔진.

핵심 원리:
- Container Envelope -> C (Format Metadata, Offsets, Checksums)
- Invariant Payload Stream -> V (CausalNodes), E (CausalEdges)
- 무손실성: 원본 바이너리 = Resynthesize(C, V, E) (100% 비트 단위 동일)
"""

from typing import Dict, List, Set, Tuple, Any, Optional, Union
import struct
import zlib

from core.topology.causal_stem_branch_engine import (
    CausalGraph, CausalNode, CausalEdge, NodeType, TrajectoryContext
)
from core.topology.base_topological_parser import (
    INVARIANT_MEM_OFFSET,
    INVARIANT_BYTE_VAL,
    INVARIANT_MEM_BLOCK
)


class BinaryContainerStripper:
    """
    구조화된 바이너리 청크 컨테이너를 해체하여 CausalGraph G = (V, E, C)로 변환하고
    임의의 수술 후 다시 완전한 규격의 바이너리로 무손실 재합성하는 엔진.
    """

    MAGIC_HEADER = b"ELYS"  # Elysian Structured Container Magic
    VERSION = 1

    def strip_and_parse(self, raw_binary: bytes) -> CausalGraph:
        """
        바이너리를 해체하여 컨테이너 규격은 C로 격리하고 데이터 스트림은 Stem으로 반환.
        """
        if len(raw_binary) < 16:
            raise ValueError("Binary buffer is too small to contain a valid container envelope.")

        # 1. 헤더 검증
        magic = raw_binary[0:4]
        if magic != self.MAGIC_HEADER:
            # 범용 바이너리로 처리
            container_type = "generic_binary"
            version = 0
            header_size = 0
            payload_bytes = raw_binary
            stored_crc = 0
        else:
            container_type = "elys_container"
            version, payload_len, stored_crc = struct.unpack(">III", raw_binary[4:16])
            header_size = 16
            payload_bytes = raw_binary[header_size:header_size + payload_len]

        graph = CausalGraph(
            graph_id="stripped_binary_graph",
            context=TrajectoryContext(
                medium_type=container_type,
                environmental_constraints={
                    "magic": magic.hex(),
                    "version": version,
                    "header_size": header_size,
                    "stored_crc": stored_crc,
                    "payload_len": len(payload_bytes)
                }
            )
        )

        # 2. 페이로드를 인과 노드(V) 및 전이 엣지(E)로 분해
        last_node_id: Optional[str] = None
        for i, byte_val in enumerate(payload_bytes):
            node_id = f"BYTE_{i}"
            node = CausalNode(
                node_id=node_id,
                node_type=NodeType.STEM,
                invariant_signature=INVARIANT_BYTE_VAL,
                payload={
                    "offset": i,
                    "val": byte_val,
                    "hex": hex(byte_val)
                }
            )
            graph.add_node(node)

            # 맥락 제약 격리 (오프셋 주소 정보는 C로 격리)
            graph.context_constraints[node_id].add(f"offset:{i}")

            if last_node_id:
                graph.add_edge(CausalEdge(
                    source_id=last_node_id,
                    target_id=node_id,
                    precondition="byte_seq_stride",
                    is_necessary=True
                ))
            last_node_id = node_id

        return graph

    def resynthesize(self, graph: CausalGraph) -> bytes:
        """
        인과 그래프 G = (V, E, C)로부터 완전한 표준 바이너리를 비트 오차 없이 재합성.
        """
        ctx = graph.context.environmental_constraints
        container_type = graph.context.medium_type

        # 1. 노드들로부터 페이로드 바이트 배열 복원
        sorted_nodes = sorted(
            [n for n in graph.nodes.values() if n.node_type == NodeType.STEM],
            key=lambda n: n.payload.get("offset", 0)
        )
        payload_bytes = bytes([n.payload["val"] for n in sorted_nodes])

        if container_type != "elys_container":
            return payload_bytes

        # 2. 컨테이너 헤더 재구성 (C에 격리되었던 규칙 및 최신 체크섬 자동 정렬)
        magic = bytes.fromhex(ctx.get("magic", self.MAGIC_HEADER.hex()))
        version = ctx.get("version", self.VERSION)
        new_payload_len = len(payload_bytes)
        new_crc = zlib.crc32(payload_bytes)

        header = magic + struct.pack(">III", version, new_payload_len, new_crc)
        return header + payload_bytes

    @classmethod
    def create_sample_container(cls, payload: bytes) -> bytes:
        """테스트 및 실증용 표준 컨테이너 바이너리 생성 헬퍼"""
        magic = cls.MAGIC_HEADER
        version = cls.VERSION
        payload_len = len(payload)
        crc = zlib.crc32(payload)
        header = magic + struct.pack(">III", version, payload_len, crc)
        return header + payload
