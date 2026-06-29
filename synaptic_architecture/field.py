import numpy as np
from scipy.ndimage import gaussian_filter

class CrystallizationField:
    """
    [Synaptic Architecture] Memristive Resistance Matrix
    Simulates the physical plasticity of a 2D memory landscape.
    Data flow (Energy) reduces resistance, creating potential wells.

    [Activation Spreading]
    The field now supports wave-like propagation of energy (Activation)
    based on local conductance (Physical paths).
    """
    def __init__(self, resolution: int = 256):
        self.resolution = resolution
        # Conductance Matrix: G = 1/R (Physical Plasticity)
        self.conductance = np.full((resolution, resolution), 0.01, dtype=np.float32)
        # Activation Matrix: Current energy flow in the field
        self.activation = np.zeros((resolution, resolution), dtype=np.float32)
        # Static Bit-Gene Map: Long-term structural storage
        self.bit_genes = np.zeros((resolution, resolution), dtype=np.uint64)

        # Thermal Control
        self.local_temperature = np.ones((resolution, resolution), dtype=np.float32)

    def inject_activation(self, pos: np.ndarray, intensity: float):
        """Injects seed energy into the field at a specific coordinate."""
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)
        self.activation[y, x] += intensity

    def propagate(self, decay: float = 0.9, spreading_factor: float = 0.5):
        """
        [Field Simultaneous Propagation]
        Spreads activation energy to neighbors based on local conductance.
        Simulates how a thought (Wave) traverses the information landscape.
        """
        # Conductance-weighted spreading
        # We use a simple convolution-like spread weighted by conductance
        # High conductance = Low resistance = Faster/Stronger spreading

        # 1. Diffusion weighted by conductance
        # This is a simplified Laplacian-based diffusion
        spread = (
            np.roll(self.activation, 1, axis=0) +
            np.roll(self.activation, -1, axis=0) +
            np.roll(self.activation, 1, axis=1) +
            np.roll(self.activation, -1, axis=1)
        ) * 0.25

        # The change in activation is proportional to conductance
        delta = (spread - self.activation) * self.conductance * spreading_factor

        # 2. Apply change and decay (Entropy)
        self.activation = (self.activation + delta) * decay

        # Ensure no negative energy
        self.activation = np.maximum(0, self.activation)

    def flow_energy(self, pos: np.ndarray, intensity: float):
        """
        [Memristive Update]
        Signal flow reinforces the conductance path (Silicon Trace).
        """
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)

        # Gaussian dissipation of conductance reinforcement
        yy, xx = np.mgrid[:self.resolution, :self.resolution]
        dist_sq = (yy - y)**2 + (xx - x)**2
        spread = 3.0 * self.local_temperature[y, x] # Temperature affects reinforcement spread

        reinforcement = (intensity * np.exp(-dist_sq / (2 * spread**2))).astype(np.float32)
        self.conductance += reinforcement

        # Hard physical limit on conductance (Saturation)
        self.conductance = np.clip(self.conductance, 0, 10.0)

    def crystallize_gene(self, pos: np.ndarray, bit_waveform: np.uint64):
        """
        Solidifies a bit-waveform into a spatial coordinate.
        """
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)
        self.bit_genes[y, x] = bit_waveform
        self.flow_energy(pos, 2.0) # Solidification is a high-energy event

    def set_local_temperature(self, pos: np.ndarray, radius: float, temp: float):
        """
        [Master's Intervention]
        Sets the temperature in a specific region of the field.
        High Temp = High Plasticity / High Search.
        Low Temp = Crystallization / Low Search.
        """
        yy, xx = np.mgrid[:self.resolution, :self.resolution]
        dist_sq = (yy - pos[0])**2 + (xx - pos[1])**2

        mask = dist_sq <= radius**2
        self.local_temperature[mask] = temp

    def apply_thermal_diffusion(self, global_entropy: float = 0.01):
        """
        Entropy: Unused paths diffuse and decay over time.
        The rate of diffusion is controlled by both global entropy and local temperature.
        """
        # Local temperature scales the diffusion sigma
        # In high-temp areas, information spreads/blurs faster
        effective_sigma = global_entropy * self.local_temperature

        # Since gaussian_filter doesn't take a 2D sigma array easily,
        # we approximate with a variable-rate decay or multiple passes.
        # For simplicity, we use the mean temperature to scale the global filter
        # but apply a local decay based on inverse temperature (High temp = higher entropy/decay)

        sigma = np.mean(effective_sigma) * 10.0
        self.conductance = gaussian_filter(self.conductance, sigma=sigma)

        # Local decay: higher temperature area decays/refreshes faster
        # (Simulating high-energy state instability)
        decay_map = 0.99 - (self.local_temperature * 0.01)
        self.conductance *= decay_map
        self.activation *= decay_map

if __name__ == "__main__":
    cf = CrystallizationField()
    cf.crystallize_gene(np.array([128, 128]), np.uint64(0xABC))
    cf.inject_activation(np.array([128, 128]), 1.0)
    cf.propagate()
    print(f"Activation at center: {cf.activation[128, 128]:.4f}")
