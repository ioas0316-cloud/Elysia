import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from core.physics.rns_multi_scale_field import ResidueNumberSystem

class TopologicalOSEngine:
    """
    [Topological & Thermodynamic OS Emulator Engine]

    A spatial, multi-scale computational manifold governed by:
    1. Toroidal Memory Geometry (via prime modular RNS distances).
    2. Phase Perturbation (Impulse Injection modifying potential energy landscape).
    3. Thermodynamic Relaxation (Langevin Dynamics with thermal fluctuations dW_t).
    4. Active Discernment:
       - Damping: Physical noise filtering and self-refusal.
       - Gradient: Natural prioritizing by potential well steepness.
       - Resonance: Task matching with existing spatial wave configurations.
    """
    def __init__(self,
                 grid_shape: Tuple[int, int] = (16, 16),
                 primes: List[int] = [5, 7, 11],
                 initial_temp: float = 10.0,
                 cooling_rate: float = 0.95,
                 damping_factor: float = 0.2,
                 diffusion_coef: float = 0.1):
        self.shape = grid_shape
        self.rns = ResidueNumberSystem(primes)
        self.primes = np.array(primes, dtype=np.int64)

        # System status arrays
        self.residues = np.ones((*grid_shape, self.rns.num_channels), dtype=np.int64) # vacuum state 1
        self.energy = np.zeros(grid_shape, dtype=np.float32)
        self.friction = np.ones(grid_shape, dtype=np.float32)

        # Thermodynamics
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.damping_factor = damping_factor
        self.diffusion_coef = diffusion_coef

        # Internal wave characteristics for Resonance matching
        # 2D phase wave representing internal state configurations (e.g. cosine/sine wave signatures)
        self.phase_waves = np.zeros(grid_shape, dtype=np.float32)
        self._init_internal_resonance_waves()

    def _init_internal_resonance_waves(self):
        """Initializes complex spatial wave patterns in the grid to serve as structural resonance reference."""
        h, w = self.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        # A complex interference pattern representing default operating frequencies/resonances of OS modules
        self.phase_waves = np.sin(2 * np.pi * y / h) * np.cos(2 * np.pi * x / w)

    def compute_toroidal_distance(self, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
        """
        Computes the toroidal Euclidean distance between two coordinates under modular RNS boundaries.
        Both dimensions wrap around seamlessly.
        """
        h, w = self.shape
        dy = abs(pt1[0] - pt2[0])
        dx = abs(pt1[1] - pt2[1])

        # Find shortest wrapped distance
        dy_wrapped = min(dy, h - dy)
        dx_wrapped = min(dx, w - dx)

        return float(np.sqrt(dy_wrapped**2 + dx_wrapped**2))

    def get_potential(self) -> np.ndarray:
        """
        Computes the potential energy landscape V(x, y).
        Higher distance of RNS residues from the ground vacuum state (1) represents higher potential tension.
        """
        p_broadcast = np.broadcast_to(self.primes, self.residues.shape)
        cw = (self.residues - 1) % p_broadcast
        ccw = (1 - self.residues) % p_broadcast
        distances = np.minimum(cw, ccw)
        return np.sum(distances, axis=-1).astype(np.float32)

    def inject_impulse(self, y: int, x: int, magnitude: float, importance: float, wave_signature: float = 1.0):
        """
        [Phase Perturbation / Impulse Injection]
        Injects a task into the system. Instead of queuing instructions, it perturbs the potential landscape
        and adds local energy.

        - importance: Controls the gradient (steepness of the potential well). High importance = fast execution.
        - wave_signature: Profile used to measure spatial resonance (Resonance Routing).
        """
        h, w = self.shape
        cy, cx = y % h, x % w

        # 1. Update energy landscape
        self.energy[cy, cx] += magnitude

        # 2. Excite RNS residues to create a modular potential well centered at (cy, cx)
        for idx, prime in enumerate(self.primes):
            # Modular excitation: perturb current residue value
            self.residues[cy, cx, idx] = (self.residues[cy, cx, idx] + int(importance)) % prime
            if self.residues[cy, cx, idx] == 1 and prime > 2:
                self.residues[cy, cx, idx] = 2  # prevent accidental fall back to vacuum immediately

        # 3. Modify internal phase wave signature by impulse wave_signature
        self.phase_waves[cy, cx] = (self.phase_waves[cy, cx] + wave_signature) / 2.0

    def step(self, dt: float = 0.1):
        """
        Advances the 위상·열역학 OS by one physical time step.
        Implements Langevin dynamics, active discernment (Damping, Gradient, Resonance), and RNS relaxation.
        """
        h, w = self.shape
        V = self.get_potential()

        # --- Active Discernment: Resonance Routing & Filtering (Vectorized) ---
        # Calculate spatial resonance with phase waves. If there's low resonance or chaotic noise,
        # apply higher damping to absorb/dissipate the energy.
        # Resonance is represented by the local gradient alignment of V and phase waves.
        mean_v = np.mean(V)
        local_resonance = np.abs(self.phase_waves * (V - mean_v))
        damping_mask = (V > 1.0) & (local_resonance < 0.1)

        # Apply intense Damping to absorb/quench noise where resonance is too low
        self.energy = np.where(damping_mask, self.energy * (1.0 - self.damping_factor * 2.0), self.energy)
        self.friction = np.where(damping_mask, 5.0, 1.0)

        # --- Langevin Dynamics (Thermodynamic Relaxation Engine) ---
        # 1. Gradient of potential field: -nabla V
        # Torus boundaries are handled by np.roll
        V_up = np.roll(V, -1, axis=0)
        V_down = np.roll(V, 1, axis=0)
        V_left = np.roll(V, -1, axis=1)
        V_right = np.roll(V, 1, axis=1)

        # 2D Laplacian (potential differences)
        laplacian = (V_up + V_down + V_left + V_right - 4 * V)

        # Flow modulated by friction (damping/resistance) and diffusion coefficient
        flow = (self.diffusion_coef * laplacian / self.friction) * dt
        self.energy += flow

        # 2. Langevin thermal fluctuation term: sqrt(2 * D * k_B * T * dt) * eta
        # Where eta is a standard normal distribution.
        # This thermal noise helps the system escape local minima and explore the global manifold.
        if self.temperature > 1e-4:
            thermal_coeff = np.sqrt(2.0 * self.diffusion_coef * self.temperature * dt)
            noise = np.random.normal(0, 1, size=self.shape) * thermal_coeff
            self.energy += noise
            # Clip energy to be non-negative
            self.energy = np.clip(self.energy, 0.0, None)

        # --- RNS Physical Relaxation ---
        # Residues step modularly closer to ground state 1
        p_broadcast = np.broadcast_to(self.primes, self.residues.shape)
        mask_not_1 = (self.residues != 1)
        cw = (self.residues - 1) % p_broadcast

        dec_mask = mask_not_1 & (cw <= (p_broadcast // 2))
        inc_mask = mask_not_1 & (~dec_mask)

        # Perform physical modular step closer to 1
        self.residues[dec_mask] = (self.residues[dec_mask] - 1) % p_broadcast[dec_mask]
        self.residues[inc_mask] = (self.residues[inc_mask] + 1) % p_broadcast[inc_mask]

        # --- Dissipation & Temperature Cooling ---
        self.energy = np.clip(self.energy - 0.05 * dt, 0.0, None)
        self.temperature *= self.cooling_rate

    def get_state(self) -> Dict[str, Any]:
        """Returns the serialized state of the Topological OS Engine."""
        return {
            "decoded_rns": self.rns.decode(self.residues).tolist(),
            "energy": self.energy.tolist(),
            "potential": self.get_potential().tolist(),
            "temperature": self.temperature,
            "resonance_profile": self.phase_waves.tolist()
        }
