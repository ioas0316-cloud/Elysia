import numpy as np
from .hardware_mapping import HardwareMemoryMap
from .bit_logic import BitInterference
from .field import CrystallizationField
from .scheduler import PCRVirtualScheduler

class DirectMappingOrganism:
    """
    [Synaptic Architecture] Hierarchical Silicon Organism
    """
    def __init__(self, resolution: int = 256):
        self.ram = HardwareMemoryMap(size=resolution * resolution)
        self.field = CrystallizationField(resolution)
        self.logic = BitInterference()
        self.scheduler = PCRVirtualScheduler()

    def flow(self, input_wave: np.uint64):
        # 1. Register Layer (Real-time Jitter)
        params = self.scheduler.get_clock_params()
        # Fix: use random bits properly for uint64
        jitter = np.uint64(np.random.randint(0, 0xFFFFFFFF, dtype=np.uint32)) | \
                 (np.uint64(np.random.randint(0, 0xFFFFFFFF, dtype=np.uint32)) << np.uint64(32))

        vibrating_wave = input_wave ^ (jitter & params['jitter_mask'])

        # 2. RAM Layer (O(1) Address Mapping)
        addr = self.ram.derive_address(vibrating_wave)
        spatial_pos = np.array([addr // self.field.resolution, addr % self.field.resolution])

        # 3. Storage Layer (Memristive Interference)
        gene = self.field.bit_genes[spatial_pos[0], spatial_pos[1]]
        resonance = self.logic.interference_score(vibrating_wave, gene)

        print(f"[Flow] Wave: {hex(vibrating_wave)} -> RAM Addr: {addr} -> Gene Res: {resonance:.4f}")

        if resonance < 0.9:
            self.field.crystallize_gene(spatial_pos, vibrating_wave)
            print(f"  > Crystallizing new Gene at {spatial_pos}")
        else:
            self.field.flow_energy(spatial_pos, 1.0)
            print(f"  > Reinforcing existing Law at {spatial_pos}")

        return spatial_pos, resonance
