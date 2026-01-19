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
from typing import List, Tuple, Optional, Any, Dict

# [CORE] Integration
try:
    from Core.Engine.Physics.core_turbine import PhotonicMonad
except ImportError:
    PhotonicMonad = Any # Placeholder

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
        self.offsets: List[int] = [] # Index of valid start positions
        self._remap()

    def _remap(self):
        """Refreshes the memory map and rebuilds the index."""
        self.file.flush()
        size = os.path.getsize(self.filepath)
        if size > 0:
            self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            self._reindex()
        else:
            self.mm = None
            self.offsets = []

    def _reindex(self):
        """Scans the file to build an offset index (O(N) but fast via mmap)."""
        self.offsets = []
        if not self.mm: return

        offset = 0
        file_size = len(self.mm)

        while offset < file_size:
            if offset + self.HEADER_SIZE > file_size: break

            # Read size from header to jump
            # Format: 7d (56) + d (8) + I (4) = 68 bytes. Size is at offset 64.
            size_bytes = self.mm[offset+64 : offset+68]
            payload_size = struct.unpack('I', size_bytes)[0]

            self.offsets.append(offset)
            offset += self.HEADER_SIZE + payload_size

    def deposit(self, vector: List[float], timestamp: float, payload: bytes) -> int:
        """
        Deposits a new experience into the sediment.
        Returns the byte offset (Address) of the deposited layer.
        """
        # Ensure vector is length 7
        if len(vector) != 7:
            vector = list(vector) + [0.0]*(7-len(vector))
            vector = vector[:7]

        # Current write position is the offset
        offset = self.file.tell()

        header = struct.pack(self.HEADER_FMT, *vector, timestamp, len(payload))
        self.file.write(header)
        self.file.write(payload)
        self.file.flush()

        # In a real high-throughput system, we wouldn't remap every write.
        # But for 'Human-Speed' interaction, it's fine.
        self._remap()

        return offset

    def store_monad(self, wavelength: float, phase: complex, intensity: float, payload: bytes) -> int:
        """
        [CORE] Stores a Photonic Monad as a Holographic Memory.
        """
        import time

        # 1. Create Holographic Vector (7D)
        vector = [
            float(wavelength),
            float(intensity),
            float(phase.real),
            float(phase.imag),
            0.0, 0.0, 0.0 # Future: Spatial Coords
        ]

        # 2. Deposit
        timestamp = time.time()
        offset = self.deposit(vector, timestamp, payload)

        logger.info(f"💎 Monad Crystalized at offset {offset} (λ={wavelength:.1e})")
        return offset

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

            if score > 0.001: # Optimization: Skip reading payload if score is 0
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
        norm_a = sum(x*x for x in vector_a) ** 0.5
        norm_b = sum(x*x for x in vector_b) ** 0.5
        return abs(norm_a - norm_b) < 0.5

    def measure_intensity(self, frequency_vector: List[float]) -> float:
        resonances = self.scan_resonance(frequency_vector, top_k=100)
        total_energy = sum(score for score, _ in resonances)
        return total_energy

    def drift(self) -> Optional[Tuple[List[float], bytes]]:
        import random
        if not self.offsets:
            return None
        offset = random.choice(self.offsets)
        return self._read_at_offset(offset)

    def glimmer(self) -> Optional[List[float]]:
        import random
        if not self.offsets:
            return None
        offset = random.choice(self.offsets)
        if not self.mm: return None
        if offset + self.HEADER_SIZE > len(self.mm): return None
        vector_bytes = self.mm[offset : offset + 56]
        vector = list(struct.unpack('7d', vector_bytes))
        return vector

    def rewind(self, steps: int = 1) -> List[Tuple[List[float], bytes]]:
        if not self.offsets:
            return []
        count = len(self.offsets)
        start_idx = max(0, count - steps)
        results = []
        for idx in range(start_idx, count):
            offset = self.offsets[idx]
            res = self._read_at_offset(offset)
            if res:
                results.append(res)
        return results

    def read_at(self, offset: int) -> Optional[Tuple[List[float], bytes]]:
        return self._read_at_offset(offset)

    def _read_at_offset(self, offset: int) -> Optional[Tuple[List[float], bytes]]:
        if not self.mm: return None
        if offset + self.HEADER_SIZE > len(self.mm): return None

        header_bytes = self.mm[offset : offset + self.HEADER_SIZE]
        data = struct.unpack(self.HEADER_FMT, header_bytes)

        vec = list(data[:7])
        payload_size = data[8]

        payload_start = offset + self.HEADER_SIZE
        payload = self.mm[payload_start : payload_start + payload_size]

        return (vec, payload)


class PrismaticSediment:
    """
    [CORE UPDATE] The Prismatic Indexer.
    Manages 7 separate SedimentLayers (Red to Violet).
    Routes memories based on their dominant vector component (Prism Color).
    """
    DOMAINS = ["RED", "ORANGE", "YELLOW", "GREEN", "BLUE", "INDIGO", "VIOLET"]

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.layers: Dict[str, SedimentLayer] = {}

        # Initialize 7 sectors
        base_dir = os.path.dirname(base_path)
        base_name = os.path.basename(base_path)

        # Ensure dir exists
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

        for i, domain in enumerate(self.DOMAINS):
            filename = f"{os.path.splitext(base_name)[0]}_{domain}.bin"
            filepath = os.path.join(base_dir, filename)
            self.layers[domain] = SedimentLayer(filepath)

        logger.info(f"🌈 PrismaticSediment initialized with 7 sectors at {base_dir}")

    def _get_dominant_domain(self, vector: List[float]) -> str:
        """Finds the index of the max value in the vector."""
        if not vector: return "RED"
        # Ensure vector is length 7
        if len(vector) < 7:
            vector = list(vector) + [0.0]*(7-len(vector))

        max_idx = np.argmax(vector[:7])
        return self.DOMAINS[max_idx]

    def deposit(self, vector: List[float], timestamp: float, payload: bytes) -> int:
        """Routes deposit to the correct sector."""
        domain = self._get_dominant_domain(vector)
        return self.layers[domain].deposit(vector, timestamp, payload)

    def store_monad(self, wavelength: float, phase: complex, intensity: float, payload: bytes) -> int:
        vector = [
            float(wavelength), float(intensity), float(phase.real), float(phase.imag),
            0.0, 0.0, 0.0
        ]
        domain = self._get_dominant_domain(vector)
        return self.layers[domain].store_monad(wavelength, phase, intensity, payload)

    def scan_resonance(self, intent_vector: List[float], top_k: int = 3) -> List[Tuple[float, bytes]]:
        domain = self._get_dominant_domain(intent_vector)
        return self.layers[domain].scan_resonance(intent_vector, top_k)

    def rewind(self, steps: int = 1) -> List[Tuple[List[float], bytes]]:
        """
        Rewinds history by merging streams from all sectors.
        To do this correctly, we must fetch the raw data (including timestamp)
        from the underlying layers, sort them, and return the payload.
        """
        all_snapshots = []

        # 1. Harvest candidates from all layers
        for layer in self.layers.values():
            # We peek directly into the offsets to get timestamps
            if not layer.offsets: continue

            # Fetch last N offsets from this layer
            count = len(layer.offsets)
            start_idx = max(0, count - steps)

            for idx in range(start_idx, count):
                offset = layer.offsets[idx]
                if not layer.mm: continue

                # Read Header to get timestamp
                header_bytes = layer.mm[offset : offset + layer.HEADER_SIZE]
                data = struct.unpack(layer.HEADER_FMT, header_bytes)

                # data = (vec[0]...vec[6], timestamp, size)
                vec = list(data[:7])
                timestamp = data[7]
                payload_size = data[8]

                # Lazily store reference
                all_snapshots.append({
                    "timestamp": timestamp,
                    "vector": vec,
                    "layer": layer,
                    "offset": offset,
                    "payload_size": payload_size
                })

        # 2. Sort by timestamp descending (Newest first)
        all_snapshots.sort(key=lambda x: x["timestamp"], reverse=True)

        # 3. Take top N
        top_n = all_snapshots[:steps]

        # 4. Fetch Payloads
        results = []
        for item in top_n:
            # We want chronological order (Oldest -> Newest) as per rewind convention?
            # Original rewind returned [t-N ... t-1] (Chronological)
            # So we should reverse this list later.

            layer = item["layer"]
            offset = item["offset"]
            size = item["payload_size"]

            payload_start = offset + layer.HEADER_SIZE
            payload = layer.mm[payload_start : payload_start + size]

            results.append((item["vector"], payload))

        # Return chronological (Oldest first)
        return list(reversed(results))

    def close(self):
        for layer in self.layers.values():
            layer.close()

    # --- Interface Compatibility ---

    def check_topology(self, vector_a: List[float], vector_b: List[float]) -> bool:
        return self.layers["RED"].check_topology(vector_a, vector_b)

    def measure_intensity(self, frequency_vector: List[float]) -> float:
        # Measure in the dominant domain
        domain = self._get_dominant_domain(frequency_vector)
        return self.layers[domain].measure_intensity(frequency_vector)

    def drift(self) -> Optional[Tuple[List[float], bytes]]:
        # Pick a random domain, then drift
        import random
        domain = random.choice(self.DOMAINS)
        return self.layers[domain].drift()

    def glimmer(self) -> Optional[List[float]]:
        import random
        domain = random.choice(self.DOMAINS)
        return self.layers[domain].glimmer()

    def read_at(self, offset: int) -> Optional[Tuple[List[float], bytes]]:
        # This is tricky. Offset is now ambiguous.
        # PrismaticSediment cannot support raw address access without a (Domain, Offset) tuple.
        # But for backward compatibility, if something calls this with a raw int,
        # it likely assumes a single file.
        # We will log a warning and fail, or try to search (too slow).
        logger.warning("⚠️ PrismaticSediment.read_at called with raw offset. This is not supported in the Fractal Index.")
        return None
