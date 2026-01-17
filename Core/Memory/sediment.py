"""
The Sediment: Unstructured Geological Memory
============================================
Core.Memory.sediment

"Don't organize. Just deposit."

This module implements Phase 5.2 (The Sediment) of the System Architecture Spec.
It replaces the concept of a Database with a 'Geological Layer'.
Data is appended as raw binary blobs to a file, and accessed via Memory Mapping (mmap).
"""

import mmap
import os
import struct
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger("Sediment")

class SedimentLayer:
    """
    A single geological layer (file).
    Format: [Vector(7 floats) | Timestamp(1 float) | Payload_Size(1 int) | Payload(bytes)]
    """
    # Header: 7 floats (vector) + 1 float (time) + 1 int (size) = 9 * 4 = 36 bytes (assuming float32/int32)
    # Actually let's use doubles for vector/time: 8 * 8 = 64 bytes. Int for size: 4 bytes.
    # Total Header = 68 bytes.
    HEADER_FMT = '7d d I' # 7 doubles (vector), 1 double (time), 1 unsigned int (size)
    HEADER_SIZE = struct.calcsize(HEADER_FMT)

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = open(filepath, "a+b")
        self.mm: Optional[mmap.mmap] = None
        self._remap()

    def _remap(self):
        """Refreshes the memory map to include new data."""
        self.file.flush()
        size = os.path.getsize(self.filepath)
        if size > 0:
            self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            self.mm = None

    def deposit(self, vector: List[float], timestamp: float, payload: bytes):
        """
        Deposits a new experience into the sediment.
        """
        # Ensure vector is length 7
        if len(vector) != 7:
            vector = list(vector) + [0.0]*(7-len(vector))
            vector = vector[:7]

        header = struct.pack(self.HEADER_FMT, *vector, timestamp, len(payload))
        self.file.write(header)
        self.file.write(payload)
        self.file.flush()

        # In a real high-throughput system, we wouldn't remap every write.
        # But for 'Human-Speed' interaction, it's fine.
        self._remap()

    def scan_resonance(self, intent_vector: List[float], top_k: int = 3) -> List[Tuple[float, bytes]]:
        """
        Scans the sediment for resonance (Cosine Similarity).
        Returns top_k (score, payload) tuples.
        """
        if not self.mm:
            return []

        intent = np.array(intent_vector, dtype=np.float64)
        intent_norm = np.linalg.norm(intent)
        if intent_norm == 0: return []

        results = []
        offset = 0
        file_size = len(self.mm)

        # Linear Scan (The "Magnet" passing over the earth)
        # In the future, this can be optimized with hierarchical indices,
        # but the spec demands "Raw Sector Access" vibe first.
        while offset < file_size:
            if offset + self.HEADER_SIZE > file_size: break

            # Read Header
            header_bytes = self.mm[offset : offset + self.HEADER_SIZE]
            data = struct.unpack(self.HEADER_FMT, header_bytes)

            vec = np.array(data[:7], dtype=np.float64)
            timestamp = data[7]
            payload_size = data[8]

            # Resonance Check (Dot Product / Cosine)
            vec_norm = np.linalg.norm(vec)
            if vec_norm > 0:
                score = np.dot(intent, vec) / (intent_norm * vec_norm)
            else:
                score = 0.0

            # Read Payload (Lazy reading: only if we need it? No, we need address)
            # For now, we store the offset/score and fetch payload later to save RAM?
            # Let's just fetch it for simplicity of the prototype.
            payload_start = offset + self.HEADER_SIZE
            payload = self.mm[payload_start : payload_start + payload_size]

            results.append((score, payload))

            offset += self.HEADER_SIZE + payload_size

        # Sort by Resonance (Descending)
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def close(self):
        if self.mm: self.mm.close()
        self.file.close()

    def check_topology(self, vector_a: List[float], vector_b: List[float]) -> bool:
        """
        [Deduction] Topological Inclusion Check.
        Does Vector Space A contain Vector Space B?
        Simplified: Is B closer to the origin of A than the boundary of A?
        """
        # Mock Logic: Simple Magnitude Comparison
        # In real space: distance(A, B) < radius(A)
        norm_a = sum(x*x for x in vector_a) ** 0.5
        norm_b = sum(x*x for x in vector_b) ** 0.5

        # A simple proxy: If A represents a larger concept, it often has higher dimensionality/magnitude
        # or serves as a centroid.
        # Here we just assume if they are close enough, they are topologically related.
        return abs(norm_a - norm_b) < 0.5

    def measure_intensity(self, frequency_vector: List[float]) -> float:
        """
        [Induction] Intensity Check.
        Measures the constructive interference of a specific frequency in the sediment.
        """
        # Scan entire sediment and sum resonance
        resonances = self.scan_resonance(frequency_vector, top_k=100)
        total_energy = sum(score for score, _ in resonances)
        return total_energy
