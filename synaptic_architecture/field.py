import numpy as np

class CrystallizationField:
    """
    [Synaptic Architecture] Memristive Crystallization Field
    A field where the 'Motion' of bits (0s and 1s) leaves permanent traces.
    Information flow (Energy) reduces resistance, creating stable synaptic canals.
    """
    def __init__(self, resolution: int = 256):
        self.resolution = resolution
        # Conductance map: High conductance = Stable Synapse (Potential Well)
        self.conductance = np.full((resolution, resolution), 0.01, dtype=np.float32)

        # Bit-Density Map: The actual vibrational content (waveforms) stored spatially
        # Represented as raw bit patterns (vibrations)
        self.bit_field = np.zeros((resolution, resolution, 64), dtype=np.uint8)

    def record_trace(self, pos: np.ndarray, signal_intensity: float):
        """
        [Memristive Trace]
        The act of information passing through a point increases its conductance.
        This is the 'Crystallization' of motion into structure.
        """
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)

        # Gaussian spread of influence (Physical interaction radius)
        yy, xx = np.mgrid[:self.resolution, :self.resolution]
        dist_sq = (yy - y)**2 + (xx - x)**2
        spread = 2.0

        # Cumulative reinforcement of the path
        influence = np.exp(-dist_sq / (2 * spread**2))
        self.conductance += (influence * signal_intensity * 0.1).astype(np.float32)
        self.conductance = np.clip(self.conductance, 0, 100.0)

    def get_motion_gradient(self, pos: np.ndarray) -> np.ndarray:
        """
        Guided by the field: Pointers slide toward higher conductance.
        'The space itself forces the mapping.'
        """
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)
        gy, gx = np.gradient(self.conductance)
        return np.array([gy[y, x], gx[y, x]])

    def solidify_bits(self, pos: np.ndarray, bit_stream: np.ndarray):
        """
        [Memory/Crystallization]
        Fixing the high-frequency motion into a stable spatial location.
        """
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)
        self.bit_field[y, x] = bit_stream
        self.record_trace(pos, 1.0) # Solidification also leaves a trace

    def apply_forgetting(self, rate: float = 0.05):
        """
        Dynamic Equilibrium: Stable paths remain, unused ones diffuse into noise.
        """
        from scipy.ndimage import gaussian_filter
        self.conductance = gaussian_filter(self.conductance, sigma=rate)

if __name__ == "__main__":
    cf = CrystallizationField()
    cf.solidify_bits(np.array([128, 128]), np.random.randint(0, 2, 64))
    print(f"Conductance at center: {cf.conductance[128, 128]}")
