import numpy as np
import mmap
import os

class TransistorGateBarrier:
    """
    [Phase: Superconducting Reality] Transistor-Gate Barrier Kernel

    This kernel implements the "Transistor = Variable Resistor" physical insight.
    It controls the flow of topological information using pure uint64 bitwise logic
    across multiple fractal scales (Bit -> Byte -> Block -> Universe).

    [Y-Delta Phase-Transition]
    - Y-Mode (Star): Reduces "cognitive inrush current" by applying a distributed sparsity filter.
      Used for initial stabilization and broad context ingestion (Start-up).
    - DELTA-Mode: Full-power superconducting discharge for maximum throughput and
      precise address extraction (Full Load).
    """

    def __init__(self, topology_path: str, scale_bits: int = 64, chunk_size: int = 1024 * 1024):
        self.topology_path = topology_path
        self.scale_bits = scale_bits
        self.chunk_size = chunk_size
        self.fd = None
        self.mmap_obj = None
        self.base_map = None
        self.dimension = 0

        # Y-Delta Mode State: Starts in Y-mode for stabilization
        self.is_delta_mode = False

        self._bind_base_conductance()

    def _bind_base_conductance(self):
        if not os.path.exists(self.topology_path):
            with open(self.topology_path, "wb") as f:
                # Initialize with random data
                f.write(np.random.randint(0, 0xFFFFFFFFFFFFFFFF, 1024 * 1024 // 8, dtype=np.uint64).tobytes())

        file_size = os.path.getsize(self.topology_path)
        self.dimension = file_size // 8
        self.fd = os.open(self.topology_path, os.O_RDONLY)
        self.mmap_obj = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)
        self.base_map = np.frombuffer(self.mmap_obj, dtype=np.uint64)
        print(f"[Barrier] Base Conductance bound: {self.dimension:,} uint64 slots mapped.")

    def set_mode_Y(self):
        """Reduces cognitive load for stabilization (Y-connection)."""
        self.is_delta_mode = False
        print("[Barrier] Mode Switched to Y (Star): Reducing Cognitive Inrush.")

    def set_mode_DELTA(self):
        """Enables full superconducting throughput (Delta-connection)."""
        self.is_delta_mode = True
        print("[Barrier] Mode Switched to DELTA: Full Causal Discharge enabled.")

    def _apply_ydelta_mask(self, gate_mask: np.uint64) -> np.uint64:
        if not self.is_delta_mode:
            # Y-mode:
            # Apply an aggressive bit-level filter to reduce cognitive load.
            # Only keep the lowest 4 bits of the intention.
            # This ensures that many values in the base_map will result in 0 when ANDed.
            return gate_mask & np.uint64(0x000000000000000F)
        return gate_mask

    def apply_gate_chunked(self, gate_mask: np.uint64):
        """
        Processes the mmap'd topology in chunks to handle TB-scale files.
        """
        adjusted_mask = self._apply_ydelta_mask(gate_mask)
        for i in range(0, self.dimension, self.chunk_size):
            end = min(i + self.chunk_size, self.dimension)
            yield self.base_map[i:end] & adjusted_mask

    def fractal_resonance(self, gate_mask: np.uint64, resolution: int = 1024) -> np.ndarray:
        """
        Observes resonance tension across the fractal field.
        """
        num_blocks = resolution
        block_size = self.dimension // num_blocks
        resonance_intensity = np.zeros(num_blocks, dtype=np.float32)

        if block_size == 0: return resonance_intensity

        for chunk_idx, chunk_flow in enumerate(self.apply_gate_chunked(gate_mask)):
            start_element = chunk_idx * self.chunk_size
            block_start_idx = start_element // block_size
            block_end_idx = (start_element + len(chunk_flow) - 1) // block_size

            for b in range(block_start_idx, min(block_end_idx + 1, num_blocks)):
                b_start = max(0, b * block_size - start_element)
                b_end = min(len(chunk_flow), (b + 1) * block_size - start_element)
                if b_start < b_end:
                    # Resonance = count of non-zero flow elements
                    resonance_intensity[b] += np.count_nonzero(chunk_flow[b_start:b_end])
        return resonance_intensity

    def reverse_discharge_interrupt(self, target_resonance: np.uint64) -> int:
        """
        [Simulation of Hardware-Level CAM Interrupt]
        Instantly targets the address of a locked resonance pattern.
        Note: Software implementation is O(N) but designed to mimic O(1) hardware discharge.
        """
        for i in range(0, self.dimension, self.chunk_size):
            end = min(i + self.chunk_size, self.dimension)
            chunk = self.base_map[i:end]
            match_indices = np.where(chunk == target_resonance)[0]
            if match_indices.size > 0:
                return i + int(match_indices[0])
        return -1

    def close(self):
        self.base_map = None
        if self.mmap_obj:
            self.mmap_obj.close()
        if self.fd:
            os.close(self.fd)

if __name__ == "__main__":
    PATH = "ydelta_poc.dat"
    if os.path.exists(PATH): os.remove(PATH)
    barrier = TransistorGateBarrier(PATH, chunk_size=1024)
    gate = np.uint64(0xFFFFFFFFFFFFFFFF)

    barrier.set_mode_Y()
    res_Y = barrier.fractal_resonance(gate, resolution=1)
    print(f"Y-Mode Resonance (Attenuated): {res_Y[0]}")

    barrier.set_mode_DELTA()
    res_D = barrier.fractal_resonance(gate, resolution=1)
    print(f"DELTA-Mode Resonance (Full): {res_D[0]}")

    barrier.close()
    if os.path.exists(PATH): os.remove(PATH)
