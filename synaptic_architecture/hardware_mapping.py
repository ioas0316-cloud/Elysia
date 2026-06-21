import numpy as np

class HardwareMemoryMap:
    def __init__(self, space_size: int = 1000000):
        self.size = space_size
        self.ram = np.zeros(space_size, dtype=np.uint64)
        self.conductance = np.zeros(space_size, dtype=np.float32)

    def derive_address(self, bitstream: np.uint64) -> int:
        """
        [O(1) Projection]
        Improved folding to reduce collisions in small address spaces.
        """
        # Using a simple hash-like fold
        x = bitstream
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        x = x ^ (x >> np.uint64(31))
        return int(x % np.uint64(self.size))

    def read_direct(self, address: int) -> np.uint64:
        return self.ram[address]

    def write_direct(self, address: int, data: np.uint64):
        self.ram[address] = data
        self.conductance[address] += 1.0
