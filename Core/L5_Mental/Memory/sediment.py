"""
The Sediment: Unstructured Geological Memory
============================================
Format: [Vector(7) | Time(1) | Size(1) | Truth(16s) | Payload]
"""

import mmap
import os
import struct
import numpy as np
import logging
from typing import *

logger = logging.getLogger("Sediment")

class DirectMemoryPointer(NamedTuple):
    sector_index: int
    byte_offset: int
    length: int

class PageAlignedAllocator:
    PAGE_SIZE = 4096
    @staticmethod
    def align_to_page(size: int) -> int:
        remainder = size % PageAlignedAllocator.PAGE_SIZE
        return 0 if remainder == 0 else PageAlignedAllocator.PAGE_SIZE - remainder

class SedimentLayer:
    HEADER_FMT = '7d d I 16s' 
    HEADER_SIZE = struct.calcsize(HEADER_FMT)

    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "a+b")
        self.mm: Optional[mmap.mmap] = None
        self.offsets: List[int] = []
        self._remap()

    def _remap(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        size = os.path.getsize(self.filepath)
        if size > 0:
            # We must close the previous mmap before reopening
            if self.mm: self.mm.close()
            # Use self.file.fileno() to mmap the actual underlying file
            self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            self._reindex()
        else:
            self.mm = None
            self.offsets = []

    def _reindex(self):
        self.offsets = []
        if not self.mm: return
        offset = 0
        file_size = len(self.mm)
        while offset < file_size:
            if offset + self.HEADER_SIZE > file_size: break
            size_bytes = self.mm[offset+64 : offset+68]
            payload_size = struct.unpack('I', size_bytes)[0]
            self.offsets.append(offset)
            block_size = self.HEADER_SIZE + payload_size
            padding = PageAlignedAllocator.align_to_page(block_size)
            offset += block_size + padding

    def deposit(self, vector: List[float], timestamp: float, payload: bytes, atomic_truth: str = "V-V-V-V-V-V-V") -> DirectMemoryPointer:
        if len(vector) != 7:
            vector = list(vector) + [0.0]*(7-len(vector))
            vector = vector[:7]
        offset = self.file.tell()
        truth_bytes = atomic_truth.encode('utf-8')[:16]
        if len(truth_bytes) < 16: truth_bytes += b'\x00' * (16 - len(truth_bytes))
        header = struct.pack(self.HEADER_FMT, *vector, timestamp, len(payload), truth_bytes)
        total_size = len(header) + len(payload)
        padding_size = PageAlignedAllocator.align_to_page(total_size)
        self.file.write(header)
        self.file.write(payload)
        self.file.write(b'\x00' * padding_size)
        self.file.flush()
        os.fsync(self.file.fileno())
        self._remap()
        return DirectMemoryPointer(offset // 4096, offset, len(payload))

    def scan_resonance(self, intent_vector: List[float], top_k: int = 3) -> List[Tuple[float, bytes, str]]:
        if not self.mm: return []
        intent = np.array(intent_vector, dtype=np.float64)
        intent_norm = np.linalg.norm(intent)
        if intent_norm == 0: return []
        results = []
        offset = 0
        while offset < len(self.mm):
            if offset + self.HEADER_SIZE > len(self.mm): break
            header = struct.unpack(self.HEADER_FMT, self.mm[offset:offset+self.HEADER_SIZE])
            vec = np.array(header[:7])
            size = header[8]
            truth = header[9].decode('utf-8').split('\0')[0]
            v_norm = np.linalg.norm(vec)
            score = np.dot(intent, vec) / (intent_norm * v_norm) if v_norm > 0 else 0.0
            payload = self.mm[offset+self.HEADER_SIZE : offset+self.HEADER_SIZE+size]
            results.append((score, payload, truth))
            padding = PageAlignedAllocator.align_to_page(self.HEADER_SIZE + size)
            offset += self.HEADER_SIZE + size + padding
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def rewind(self, steps: int = 1) -> List[Tuple[List[float], bytes, str]]:
        if not self.offsets: return []
        results = []
        for off in self.offsets[-steps:]:
            header = struct.unpack(self.HEADER_FMT, self.mm[off:off+self.HEADER_SIZE])
            payload = self.mm[off+self.HEADER_SIZE : off+self.HEADER_SIZE+header[8]]
            truth = header[9].decode('utf-8').split('\0')[0]
            results.append((list(header[:7]), payload, truth))
        return results

    def close(self):
        if self.mm: self.mm.close()
        self.file.close()
