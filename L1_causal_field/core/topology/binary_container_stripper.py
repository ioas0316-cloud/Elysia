"""
Elysia Binary Container Stripper Module - PNG Causal Parser & Reassembler
========================================================================
공개 규격인 PNG (Portable Network Graphics - RFC 2083 및 W3C Specification) 헤더 및 청크 구조를
파이썬 표준 라이브러리(struct, zlib)를 사용하여 파싱하고 무손실 재합성(Reassemble)하는 비트-구조 1:1 매핑 파서.

PNG 구조:
- PNG Signature: 8바이트 (89 50 4E 47 0D 0A 1A 0A)
- Chunk Structure:
  - Length (4 bytes, Big-Endian uint32)
  - Chunk Type (4 bytes ASCII e.g. IHDR, IDAT, PLTE, IEND)
  - Chunk Data (Length bytes)
  - CRC32 (4 bytes, uint32, calculated over Chunk Type + Chunk Data)
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import struct
import zlib

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

class PNGChunk:
    def __init__(self, chunk_type: bytes, length: int, data: bytes, crc: int):
        self.chunk_type = chunk_type  # e.g., b'IHDR'
        self.length = length
        self.data = data
        self.crc = crc

    def verify_crc(self) -> bool:
        calculated = zlib.crc32(self.chunk_type + self.data) & 0xffffffff
        return calculated == self.crc

    def to_bytes(self) -> bytes:
        calculated_crc = zlib.crc32(self.chunk_type + self.data) & 0xffffffff
        return struct.pack(">I", self.length) + self.chunk_type + self.data + struct.pack(">I", calculated_crc)


from core.topology.causal_stem_branch_engine import (
    CausalGraph, CausalNode, CausalEdge, NodeType, TrajectoryContext
)


class BinaryContainerStripper:
    """
    바이너리 컨테이너 껍데기를 박리하여 순수 인과 줄기(Stem) 노드 그래프로 변환하고
    100% 무손실 재합성을 수행하는 박리 엔진.
    """

    def create_sample_container(self, payload: bytes) -> bytes:
        crc = zlib.crc32(payload) & 0xffffffff
        header = struct.pack(">4sI4sI", b"ELYS", 1, b"PAD\x00", crc)
        return header + payload

    def strip_and_parse(self, raw_binary: bytes) -> CausalGraph:
        if len(raw_binary) < 16:
            raise ValueError("Binary container too short for header")

        magic, ver, _, stored_crc = struct.unpack(">4sI4sI", raw_binary[:16])
        payload = raw_binary[16:]

        graph = CausalGraph(
            graph_id="binary_container_graph",
            context=TrajectoryContext(
                medium_type="elys_container",
                environmental_constraints={
                    "magic": magic.decode("latin1", errors="ignore").rstrip("\x00"),
                    "version": ver,
                    "stored_crc": stored_crc
                }
            )
        )

        for i, b in enumerate(payload):
            node_id = f"BYTE_{i}"
            node = CausalNode(
                node_id=node_id,
                node_type=NodeType.STEM,
                invariant_signature="INVARIANT_BYTE_VAL",
                payload={"val": b, "idx": i, "hex": hex(b)}
            )
            graph.add_node(node)
            if i > 0:
                prev_id = f"BYTE_{i-1}"
                graph.add_edge(CausalEdge(
                    source_id=prev_id,
                    target_id=node_id,
                    precondition="byte_sequence_continuity"
                ))

        return graph

    def resynthesize(self, graph: CausalGraph) -> bytes:
        byte_nodes = [
            n for n in graph.nodes.values()
            if n.node_id.startswith("BYTE_")
        ]
        byte_nodes.sort(key=lambda n: n.payload.get("idx", int(n.node_id.split("_")[1])))

        payload_bytes = bytes([n.payload["val"] for n in byte_nodes])
        crc = zlib.crc32(payload_bytes) & 0xffffffff

        ver = graph.context.environmental_constraints.get("version", 1)
        magic_str = graph.context.environmental_constraints.get("magic", "ELYS")
        magic_bytes = magic_str.encode("latin1")[:4].ljust(4, b"\x00")

        header = struct.pack(">4sI4sI", magic_bytes, ver, b"PAD\x00", crc)
        return header + payload_bytes


class PNGContainerStripper:
    """
    PNG 바이너리 청크 컨테이너를 완벽하게 파싱하여 구조화하고
    재합성(Reassemble)을 통해 비트 오차 없이 원본을 복원하는 1:1 매핑 엔진.
    """

    PROVENANCE_REAL_DERIVED = "REAL_DERIVED"

    def parse(self, raw_png: bytes) -> Dict[str, Any]:
        """
        PNG 바이너리를 공식 규격에 따라 시그니처와 청크 단위로 파싱
        """
        if len(raw_png) < 8 or raw_png[:8] != PNG_SIGNATURE:
            raise ValueError("Invalid PNG Signature")

        signature = raw_png[:8]
        offset = 8
        chunks: List[PNGChunk] = []

        while offset < len(raw_png):
            if offset + 8 > len(raw_png):
                raise ValueError(f"Truncated PNG chunk header at offset {offset}")

            length, chunk_type = struct.unpack(">I4s", raw_png[offset:offset+8])
            offset += 8

            if offset + length + 4 > len(raw_png):
                raise ValueError(f"Truncated PNG chunk data at offset {offset}")

            data = raw_png[offset:offset+length]
            offset += length

            crc = struct.unpack(">I", raw_png[offset:offset+4])[0]
            offset += 4

            chunk = PNGChunk(chunk_type=chunk_type, length=length, data=data, crc=crc)
            if not chunk.verify_crc():
                raise ValueError(f"CRC Mismatch in chunk {chunk_type.decode('latin1', errors='ignore')}")

            chunks.append(chunk)

        return {
            "signature": signature,
            "chunks": chunks,
            "provenance": self.PROVENANCE_REAL_DERIVED
        }

    def reassemble(self, parsed_structure: Dict[str, Any]) -> bytes:
        """
        파싱된 PNG 청크 구조체로부터 원본 파일 바이너리를 100% 비트 오차 없이 재합성
        """
        signature: bytes = parsed_structure["signature"]
        chunks: List[PNGChunk] = parsed_structure["chunks"]

        output = bytearray(signature)
        for chunk in chunks:
            output.extend(chunk.to_bytes())

        return bytes(output)
