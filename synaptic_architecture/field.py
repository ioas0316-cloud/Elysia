import numpy as np

class MemristiveField:
    """
    [Synaptic Architecture] Memristive Potential Field
    A pure physical field where information flow alters the spatial resistance (conductance).
    No discrete addresses exist; only the gradient of the potential field guides the pointers.
    """
    def __init__(self, resolution: int = 256):
        self.resolution = resolution
        # Conductance map: High conductance = Low resistance (Potential Well)
        # Initialized with a very low base conductance (High resistance)
        self.conductance = np.full((resolution, resolution), 0.01, dtype=np.float32)

        # Data Field: The actual vibrational content stored at each spatial point
        self.data_field = np.zeros((resolution, resolution, 64), dtype=np.float32)

    def propagate_signal(self, pos: np.ndarray, intensity: float):
        """
        Leaving a trace (Memristivity): Signal flow creates a 'conductance canal'.
        """
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)

        # Create a Gaussian update (Diffusion of Conductance)
        # This simulates the physical 'widening' of the synaptic path
        yy, xx = np.mgrid[:self.resolution, :self.resolution]
        dist_sq = (yy - y)**2 + (xx - x)**2
        spread = 3.0
        # Conductance update (Log-Normal or similar cumulative plasticity)
        self.conductance += (intensity * np.exp(-dist_sq / (2 * spread**2))).astype(np.float32)

    def get_potential_gradient(self, pos: np.ndarray) -> np.ndarray:
        """
        Calculate the local gradient of the conductance field.
        A pointer 'slides' towards higher conductance (Potential Well).
        """
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)

        # Use central difference for gradient
        gy, gx = np.gradient(self.conductance)
        return np.array([gy[y, x], gx[y, x]])

    def deposit_engram(self, pos: np.ndarray, engram: np.ndarray):
        """
        Deposit vibrational data into the field.
        This also triggers a memristive conductance update.
        """
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)
        self.data_field[y, x] = engram
        self.propagate_signal(pos, 2.0) # Data deposition creates strong initial conductance

    def apply_diffusion(self, rate: float = 0.01):
        """
        Entropy/Forgetfulness: Over time, the conductance canals blur unless reinforced.
        """
        from scipy.ndimage import gaussian_filter
        self.conductance = gaussian_filter(self.conductance, sigma=rate)

if __name__ == "__main__":
    mf = MemristiveField()
    mf.deposit_engram(np.array([128, 128]), np.random.randn(64))
    print(f"Gradient at (120, 120): {mf.get_potential_gradient(np.array([120, 120]))}")
