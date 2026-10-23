r"""
.CST (Causal Spatiotemporal Tensor) Serialization Container Handler
==================================================================

Handles zero-copy binary serialization / deserialization of causal spatiotemporal streams:
- Header (Magic 'CST1', Version, Num_Nodes, Dim, Axiom Hash)
- Topological Anchor Table
- I-Frame Baseline Blocks
- Delta Stream P-Frame Chunks
"""

import struct
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional


class CSTContainerHandler:
    """
    Binary reader / writer for .CST (Causal Spatiotemporal Tensor) files.
    """

    MAGIC = b"CST1"  # Magic Bytes

    def __init__(self, filename: str):
        self.filename = filename

    def write_container(
        self,
        num_nodes: int,
        dim: int,
        topo_weight: np.ndarray,
        topo_relation: np.ndarray,
        i_frames: List[np.ndarray],
        p_frame_deltas: List[Tuple[float, np.ndarray]]
    ):
        """
        Serializes causal spatiotemporal stream into binary .CST format.

        Header Format:
        - Magic (4 bytes)
        - Version (uint32)
        - Num Nodes (uint32)
        - Dim (uint32)
        - Num I-Frames (uint32)
        - Num P-Frames (uint32)
        """
        with open(self.filename, "wb") as f:
            # 1. Header Write
            version = 1
            num_i_frames = len(i_frames)
            num_p_frames = len(p_frame_deltas)

            header_bin = struct.pack(
                "<4sIIIII",
                self.MAGIC,
                version,
                num_nodes,
                dim,
                num_i_frames,
                num_p_frames
            )
            f.write(header_bin)

            # 2. Topological Anchor Table (Weight & Relation)
            f.write(topo_weight.astype(np.float32).tobytes())
            f.write(topo_relation.astype(np.float32).tobytes())

            # 3. I-Frame Blocks
            for iframe in i_frames:
                f.write(iframe.astype(np.float32).tobytes())

            # 4. P-Frame Delta Blocks (Timestamp float64 + Delta float32 array)
            for timestamp, delta in p_frame_deltas:
                f.write(struct.pack("<d", float(timestamp)))
                f.write(delta.astype(np.float32).tobytes())

    def read_container(self) -> Dict[str, Any]:
        """
        Deserializes binary .CST file into structured dictionaries and NumPy arrays.
        """
        with open(self.filename, "rb") as f:
            header_size = struct.calcsize("<4sIIIII")
            header_data = f.read(header_size)
            if len(header_data) < header_size:
                raise ValueError("Invalid .CST file header: File too short.")

            magic, version, num_nodes, dim, num_i_frames, num_p_frames = struct.unpack("<4sIIIII", header_data)

            if magic != self.MAGIC:
                raise ValueError(f"Invalid .CST magic bytes: {magic}")

            # Topological Anchor Table
            node_matrix_bytes = num_nodes * num_nodes * 4
            topo_w_bytes = f.read(node_matrix_bytes)
            topo_r_bytes = f.read(node_matrix_bytes)

            topo_weight = np.frombuffer(topo_w_bytes, dtype=np.float32).reshape(num_nodes, num_nodes)
            topo_relation = np.frombuffer(topo_r_bytes, dtype=np.float32).reshape(num_nodes, num_nodes)

            # I-Frame Blocks
            iframe_bytes_len = num_nodes * dim * 4
            i_frames = []
            for _ in range(num_i_frames):
                raw = f.read(iframe_bytes_len)
                i_frames.append(np.frombuffer(raw, dtype=np.float32).reshape(num_nodes, dim))

            # P-Frame Blocks
            p_frame_deltas = []
            pframe_bytes_len = num_nodes * dim * 4
            for _ in range(num_p_frames):
                ts_bytes = f.read(8)
                (timestamp,) = struct.unpack("<d", ts_bytes)
                raw_delta = f.read(pframe_bytes_len)
                delta = np.frombuffer(raw_delta, dtype=np.float32).reshape(num_nodes, dim)
                p_frame_deltas.append((timestamp, delta))

            return {
                "version": version,
                "num_nodes": num_nodes,
                "dim": dim,
                "topo_weight": topo_weight,
                "topo_relation": topo_relation,
                "i_frames": i_frames,
                "p_frame_deltas": p_frame_deltas
            }
