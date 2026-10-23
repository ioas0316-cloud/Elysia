import numpy as np

class RawBitField:
    def __init__(self, size: int = 10000):
        self.size = size
        self.conductance = np.zeros(size, dtype=np.float32)
        self.data = np.zeros(size, dtype=np.uint64)

    def trace_vibration(self, index: int, intensity: float = 100.0, radius: int = 1000):
        start = max(0, index - radius)
        end = min(self.size, index + radius)
        for i in range(start, end):
            dist = abs(i - index)
            strength = intensity * (1.0 - (dist / radius))
            if strength > self.conductance[i]:
                self.conductance[i] = strength

    def get_local_gradient(self, index: int) -> int:
        idx = int(np.clip(index, 1, self.size - 2))
        left = self.conductance[idx - 1]
        right = self.conductance[idx + 1]
        curr = self.conductance[idx]

        if right > curr: return 1
        if left > curr: return -1
        return 0

    def solidify(self, index: int, bitstream: np.uint64):
        idx = int(np.clip(index, 0, self.size - 1))
        self.data[idx] = bitstream
        # Also increment neighbor data slightly to provide "resonance gradients"
        radius = 50
        for i in range(max(0, idx-radius), min(self.size, idx+radius)):
            if self.data[i] == 0:
                self.data[i] = bitstream ^ np.uint64(np.random.randint(0, 0xFFFFFFFF))

        self.data[idx] = bitstream # Ensure center is exact
        self.trace_vibration(idx)
