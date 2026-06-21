import numpy as np
from .hardware_mapping import HardwareMemoryMap
from .causal_gene import CausalGeneMap

class DirectMappingOrganism:
    """
    [Synaptic Architecture] Direct Mapping Evolution
    Abolishes search. The bitstream determines its own address.
    1. Perception: Wave ingestion.
    2. Mapping: O(1) Address Projection.
    3. Interference: Hardware Bit-Masking.
    4. Re-cognition: Synaptic stabilization.
    5. Memory: Gene Crystallization.
    """
    def __init__(self, size: int = 1000000):
        self.ram = HardwareMemoryMap(size)
        self.genes = CausalGeneMap()

    def flow(self, input_wave: np.uint64):
        print(f"\n[Direct Evolution] Input: {hex(input_wave)}")

        # 1. Perception
        # (Jitter would happen here on the time axis)

        # 2. Mapping (O(1) Address Derivation)
        addr = self.ram.derive_address(input_wave)
        print(f" 2. Mapping: Projected to RAM Address {addr}")

        # 3. Interference (Hardware Filter)
        matches = self.genes.hardware_filter(input_wave)
        if matches:
            print(f" 3. Interference: Resonating with Genes: {matches}")
        else:
            print(" 3. Interference: No existing structural resonance.")

        # 4. Re-cognition / Memory
        # Write directly to the projected address
        self.ram.write_direct(addr, input_wave)
        print(f" 4 & 5. Recognition/Memory: Crystallized in RAM at {addr}")

        return addr

if __name__ == "__main__":
    dmo = DirectMappingOrganism()
    dmo.flow(np.uint64(0xFEEDFACEBEEFCAFE))
