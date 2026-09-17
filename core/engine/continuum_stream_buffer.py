"""Continuum Stream Buffer for Elysia Engine.

Provides a memory-mapped (mmap) rolling ring buffer for high-density,
contiguous binary and wave stream storage. Rejects heap pointer/malloc
fragmentation in favor of continuous byte trajectory tracking.
"""

import os
import mmap
import tempfile
from typing import Union, Tuple, Optional
import numpy as np


class ContinuumStreamBuffer:
    """A memory-mapped continuous rolling ring buffer for wave/frequency stream processing."""

    def __init__(
        self,
        capacity_bytes: int = 10 * 1024 * 1024,  # Default 10 MB for testing / configurable
        filepath: Optional[str] = None,
        auto_cleanup: bool = True
    ):
        self.capacity_bytes = capacity_bytes
        self.auto_cleanup = auto_cleanup
        self.total_bytes_written = 0
        self.head_position = 0

        if filepath is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, prefix="elysia_continuum_", suffix=".bin")
            self.filepath = temp_file.name
            temp_file.close()
        else:
            self.filepath = filepath

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)

        # Allocate file space and create mmap
        with open(self.filepath, "wb") as f:
            f.seek(self.capacity_bytes - 1)
            f.write(b"\x00")
            f.flush()

        self._file_obj = open(self.filepath, "r+b")
        self.mmap_obj = mmap.mmap(self._file_obj.fileno(), self.capacity_bytes, access=mmap.ACCESS_WRITE)

    def write_stream(self, data: Union[bytes, bytearray, np.ndarray]) -> Tuple[int, int]:
        """Writes raw byte stream into the rolling mmap buffer without allocation fragmentation.

        Returns:
            Tuple[int, int]: (sequence_start_index, bytes_written)
        """
        if isinstance(data, np.ndarray):
            raw_bytes = data.tobytes()
        elif isinstance(data, (bytes, bytearray)):
            raw_bytes = bytes(data)
        else:
            raise TypeError(f"Unsupported data type for stream buffer: {type(data)}")

        length = len(raw_bytes)
        if length == 0:
            return (self.total_bytes_written, 0)

        seq_start = self.total_bytes_written

        if length >= self.capacity_bytes:
            # Data exceeds entire capacity; keep only the last capacity_bytes
            raw_bytes = raw_bytes[-self.capacity_bytes:]
            length = len(raw_bytes)
            self.mmap_obj[:length] = raw_bytes
            self.head_position = 0
        else:
            tail_space = self.capacity_bytes - self.head_position
            if length <= tail_space:
                self.mmap_obj[self.head_position : self.head_position + length] = raw_bytes
                self.head_position = (self.head_position + length) % self.capacity_bytes
            else:
                part1 = raw_bytes[:tail_space]
                part2 = raw_bytes[tail_space:]
                self.mmap_obj[self.head_position : self.capacity_bytes] = part1
                self.mmap_obj[0 : len(part2)] = part2
                self.head_position = len(part2)

        self.total_bytes_written += length
        return (seq_start, length)

    def read_recent(self, length: int) -> bytes:
        """Reads the most recent N bytes written to the rolling continuum buffer."""
        read_len = min(length, self.capacity_bytes, self.total_bytes_written)
        if read_len == 0:
            return b""

        end_pos = self.head_position
        start_pos = (end_pos - read_len) % self.capacity_bytes

        if start_pos < end_pos:
            return bytes(self.mmap_obj[start_pos:end_pos])
        else:
            part1 = self.mmap_obj[start_pos : self.capacity_bytes]
            part2 = self.mmap_obj[0:end_pos]
            return bytes(part1) + bytes(part2)

    def read_numpy_recent(self, length: int, dtype=np.uint8) -> np.ndarray:
        """Reads recent buffer content as a 1D numpy array."""
        raw = self.read_recent(length)
        return np.frombuffer(raw, dtype=dtype)

    def read_range(self, seq_start: int, length: int) -> bytes:
        """Reads a specific sequence range from the buffer if still within rolling history."""
        if seq_start + length <= self.total_bytes_written - self.capacity_bytes:
            raise ValueError("Requested stream range has been overwritten in rolling buffer window")

        offset_from_start = seq_start % self.capacity_bytes
        if offset_from_start + length <= self.capacity_bytes:
            return bytes(self.mmap_obj[offset_from_start : offset_from_start + length])
        else:
            part1_len = self.capacity_bytes - offset_from_start
            part2_len = length - part1_len
            part1 = self.mmap_obj[offset_from_start : self.capacity_bytes]
            part2 = self.mmap_obj[0:part2_len]
            return bytes(part1) + bytes(part2)

    def flush(self):
        """Flushes memory map changes to physical storage."""
        if self.mmap_obj:
            self.mmap_obj.flush()

    def close(self):
        """Closes memory map and underlying file resource."""
        if hasattr(self, "mmap_obj") and self.mmap_obj is not None:
            self.mmap_obj.close()
            self.mmap_obj = None
        if hasattr(self, "_file_obj") and self._file_obj is not None:
            self._file_obj.close()
            self._file_obj = None

        if self.auto_cleanup and os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except OSError:
                pass

    def __del__(self):
        self.close()
