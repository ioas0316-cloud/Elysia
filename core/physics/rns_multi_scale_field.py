import numpy as np
from typing import Dict, List, Tuple, Optional, Any

def ext_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean Algorithm. Returns (gcd, x, y) such that ax + by = gcd."""
    if a == 0:
        return b, 0, 1
    g, x1, y1 = ext_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return g, x, y

def modular_inverse(a: int, m: int) -> int:
    """Computes the modular inverse of a modulo m."""
    g, x, _ = ext_gcd(a, m)
    if g != 1:
        raise ValueError(f"Modular inverse of {a} mod {m} does not exist (not coprime).")
    return x % m

class ResidueNumberSystem:
    """
    [Residue Number System (RNS) Processor]
    Performs exact, carry-free parallel modular arithmetic on integer states.
    Uses the Chinese Remainder Theorem (CRT) for lossless reconstruction of states.
    """
    def __init__(self, primes: List[int]):
        self.primes = np.array(primes, dtype=np.int64)
        self.num_channels = len(primes)

        # Ensure pairwise coprimality
        for i in range(len(primes)):
            for j in range(i + 1, len(primes)):
                if np.gcd(primes[i], primes[j]) != 1:
                    raise ValueError(f"Primes must be pairwise coprime. Got {primes[i]} and {primes[j]}.")

        # Dynamic range M = prod(p_i)
        self.M = int(np.prod(self.primes))

        # CRT precomputations
        self.M_i = np.array([self.M // p for p in primes], dtype=np.int64)
        self.N_i = np.array([modular_inverse(int(self.M_i[idx]), int(primes[idx])) for idx in range(len(primes))], dtype=np.int64)

        # Combined CRT multiplier factors mod M
        self.crt_factors = (self.M_i * self.N_i) % self.M

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encodes an integer array into RNS residues of shape (..., num_channels)."""
        x_expanded = np.expand_dims(x, axis=-1)
        return x_expanded % self.primes

    def decode(self, residues: np.ndarray) -> np.ndarray:
        """Decodes an RNS residue array of shape (..., num_channels) back to integers in [0, M-1]."""
        # residues * crt_factors along the last dimension
        summed = np.sum(residues * self.crt_factors, axis=-1)
        return summed % self.M

    def add(self, res_a: np.ndarray, res_b: np.ndarray) -> np.ndarray:
        """Carry-free addition of two residue arrays."""
        return (res_a + res_b) % self.primes

    def subtract(self, res_a: np.ndarray, res_b: np.ndarray) -> np.ndarray:
        """Carry-free subtraction of two residue arrays."""
        return (res_a - res_b) % self.primes

    def multiply(self, res_a: np.ndarray, res_b: np.ndarray) -> np.ndarray:
        """Carry-free multiplication of two residue arrays."""
        return (res_a * res_b) % self.primes


class MicroGrid:
    """
    [Micro-Scale Local High-Resolution Grid]
    Active local grid representing a zoom-in state under high-energy stimulation.
    Uses larger primes to achieve fine-grained physical wave simulation.
    """
    def __init__(self, parent_pos: Tuple[int, int], sub_shape: Tuple[int, int], rns: ResidueNumberSystem):
        self.parent_pos = parent_pos  # (y, x) in parent macro grid
        self.shape = sub_shape
        self.rns = rns

        # Residues initialized to ground state 1
        self.residues = np.ones((*sub_shape, rns.num_channels), dtype=np.int64)
        # Energy initialized to neutral
        self.energy = np.zeros(sub_shape, dtype=np.float32)
        # Friction/resistance map
        self.friction = np.ones(sub_shape, dtype=np.float32)

    def initialize_potential_well(self, base_val: int):
        """Initializes a smooth quadratic physical potential well around the center of the micro-grid."""
        h, w = self.shape
        cy, cx = h // 2, w // 2
        for y in range(h):
            for x in range(w):
                dist_sq = (y - cy)**2 + (x - cx)**2
                # Quadratic potential well around base value
                val = (base_val + dist_sq) % self.rns.M
                self.residues[y, x] = self.rns.encode(np.array(val))


class MultiScaleRNSField:
    """
    [Multi-Scale Residue Number System Field]
    Adapts spatial scale and prime modulos based on system excitation and local energy.
    Integrates Torus Topology, Self-Outpouring, Variable Resistance Friction,
    and Physical Relaxation into the Multi-Scale Prime Hierarchy.
    """
    def __init__(self,
                 macro_shape: Tuple[int, int] = (16, 16),
                 micro_shape: Tuple[int, int] = (4, 4),
                 macro_primes: List[int] = [3, 5, 7],
                 micro_primes: List[int] = [11, 13, 17],
                 zoom_threshold: float = 10.0,
                 decay_threshold: float = 1.0,
                 dissipation_rate: float = 0.05):
        self.macro_shape = macro_shape
        self.micro_shape = micro_shape

        self.macro_rns = ResidueNumberSystem(macro_primes)
        self.micro_rns = ResidueNumberSystem(micro_primes)

        self.zoom_threshold = zoom_threshold
        self.decay_threshold = decay_threshold
        self.dissipation_rate = dissipation_rate

        # Initialize Macro Grid residues to 1 (Vacuum State)
        self.macro_residues = np.ones((*macro_shape, self.macro_rns.num_channels), dtype=np.int64)
        self.macro_energy = np.zeros(macro_shape, dtype=np.float32)
        self.macro_friction = np.ones(macro_shape, dtype=np.float32)

        # Active micro grids mapped by parent coordinate (y, x)
        self.micro_grids: Dict[Tuple[int, int], MicroGrid] = {}

    def get_macro_potential(self) -> np.ndarray:
        """
        Computes potential of each macro cell based on residue distance to ground state 1.
        Higher distance to 1 implies higher system tension (Excitement).
        """
        # Distances along the torus circles
        p_broadcast = np.broadcast_to(self.macro_rns.primes, self.macro_residues.shape)
        cw = (self.macro_residues - 1) % p_broadcast
        ccw = (1 - self.macro_residues) % p_broadcast
        distances = np.minimum(cw, ccw)
        # Sum distances over all prime channels to represent overall potential tension
        return np.sum(distances, axis=-1).astype(np.float32)

    def get_micro_potential(self, mgrid: MicroGrid) -> np.ndarray:
        """Computes modular potential of each micro cell inside a MicroGrid."""
        p_broadcast = np.broadcast_to(mgrid.rns.primes, mgrid.residues.shape)
        cw = (mgrid.residues - 1) % p_broadcast
        ccw = (1 - mgrid.residues) % p_broadcast
        distances = np.minimum(cw, ccw)
        return np.sum(distances, axis=-1).astype(np.float32)

    def stimulate(self, y: int, x: int, energy_amount: float):
        """Injects external energy stimulation into a macro cell, causing excitation."""
        h, w = self.macro_shape
        # Ensure toroidal coordinate wrap-around
        y_wrap, x_wrap = y % h, x % w
        self.macro_energy[y_wrap, x_wrap] += energy_amount

        # Excite the RNS residues at that location (move them away from 1)
        # We perform a deterministic perturbation modulo primes to represent excitement
        current_res = self.macro_residues[y_wrap, x_wrap]
        excited_res = (current_res + 1) % self.macro_rns.primes
        # Make sure we don't accidentally land on 1 if we were excited; if so, push to 2
        for idx, prime in enumerate(self.macro_rns.primes):
            if excited_res[idx] == 1 and prime > 2:
                excited_res[idx] = 2
        self.macro_residues[y_wrap, x_wrap] = excited_res

    def step(self, dt: float = 0.1):
        """
        Advances the field by one step.
        1. Evaluates macro cells for zoom-in (spawning high-res MicroGrids).
        2. Simulates potential flow (Self-Outpouring) under Torus boundary and variable friction.
        3. Simulates physical relaxation towards the ground state 1.
        4. Collapses and renormalizes low-energy MicroGrids back to macro scale.
        """
        # Step 1: Manage Zoom-In / Spawning
        for y in range(self.macro_shape[0]):
            for x in range(self.macro_shape[1]):
                if self.macro_energy[y, x] >= self.zoom_threshold and (y, x) not in self.micro_grids:
                    # Spawn high-resolution micro-grid
                    mgrid = MicroGrid((y, x), self.micro_shape, self.micro_rns)

                    # Project current macro-state into the micro-grid using a potential well centered at decoded value
                    decoded_macro_val = self.macro_rns.decode(self.macro_residues[y, x])
                    mgrid.initialize_potential_well(decoded_macro_val)

                    # Divide energy between macro and micro grid
                    mgrid.energy[:] = self.macro_energy[y, x] / 2.0
                    self.macro_energy[y, x] /= 2.0

                    self.micro_grids[(y, x)] = mgrid

        # Step 2: Self-Outpouring Flow (Laplacian Diffusion on a Torus)
        # Macro Grid Potential & Energy diffusion
        V_macro = self.get_macro_potential()

        V_up = np.roll(V_macro, -1, axis=0)
        V_down = np.roll(V_macro, 1, axis=0)
        V_left = np.roll(V_macro, -1, axis=1)
        V_right = np.roll(V_macro, 1, axis=1)

        # Laplacian represents potential differences (toroidal boundaries handle wrap automatically)
        macro_laplacian = (V_up + V_down + V_left + V_right - 4 * V_macro)

        # Energy flow modulated by variable friction
        macro_flow = (macro_laplacian / self.macro_friction) * dt
        self.macro_energy += macro_flow
        self.macro_energy = np.clip(self.macro_energy - self.dissipation_rate * dt, 0.0, None)

        # Excite RNS residues where energy flows/increases
        excitation_mask = (macro_flow > 0.1)
        if np.any(excitation_mask):
            self.macro_residues[excitation_mask] = (self.macro_residues[excitation_mask] + 1) % self.macro_rns.primes

        # Micro Grid internal flow
        for (my, mx), mgrid in list(self.micro_grids.items()):
            V_micro = self.get_micro_potential(mgrid)

            # 2D diffusion inside micro grid (with reflecting boundaries since it is localized)
            m_up = np.roll(V_micro, -1, axis=0); m_up[-1, :] = V_micro[-1, :]
            m_down = np.roll(V_micro, 1, axis=0); m_down[0, :] = V_micro[0, :]
            m_left = np.roll(V_micro, -1, axis=1); m_left[:, -1] = V_micro[:, -1]
            m_right = np.roll(V_micro, 1, axis=1); m_right[:, 0] = V_micro[:, 0]

            micro_laplacian = (m_up + m_down + m_left + m_right - 4 * V_micro)
            micro_flow = (micro_laplacian / mgrid.friction) * dt
            mgrid.energy += micro_flow
            mgrid.energy = np.clip(mgrid.energy - self.dissipation_rate * dt, 0.0, None)

            # Local micro-excitation
            m_excite = (micro_flow > 0.1)
            if np.any(m_excite):
                mgrid.residues[m_excite] = (mgrid.residues[m_excite] + 1) % mgrid.rns.primes

        # Step 3: Physical Relaxation (Falling into Ground State 1)
        # Macro Relaxation
        self.macro_residues = self._relax_residue_array(self.macro_residues, self.macro_rns.primes)

        # Micro Relaxation
        for mgrid in self.micro_grids.values():
            mgrid.residues = self._relax_residue_array(mgrid.residues, mgrid.rns.primes)

        # Step 4: Renormalization & Zoom-Out
        to_remove = []
        for (my, mx), mgrid in self.micro_grids.items():
            total_micro_energy = float(np.sum(mgrid.energy))
            if total_micro_energy < self.decay_threshold:
                # Merge back to parent macro cell (Renormalization Group coalescing)
                to_remove.append((my, mx))

                # Coalesce micro energies to parent
                self.macro_energy[my, mx] += total_micro_energy

                # Average residue representation: decode micro cells, take mean modulo macro M
                decoded_vals = mgrid.rns.decode(mgrid.residues)
                mean_val = int(np.mean(decoded_vals)) % self.macro_rns.M

                # Update parent RNS residues
                self.macro_residues[my, mx] = self.macro_rns.encode(np.array(mean_val))

        for key in to_remove:
            del self.micro_grids[key]

    def _relax_residue_array(self, residues: np.ndarray, primes: np.ndarray) -> np.ndarray:
        """
        [Modular Gradient Descent]
        Independently steps residues one step closer to the ground state 1 on each GF(p) circle.
        """
        p_broadcast = np.broadcast_to(primes, residues.shape)
        new_residues = residues.copy()

        # Shortest path check
        mask_not_1 = (residues != 1)
        cw = (residues - 1) % p_broadcast

        # Decrement clockwise path, increment counter-clockwise path
        dec_mask = mask_not_1 & (cw <= (p_broadcast // 2))
        inc_mask = mask_not_1 & (~dec_mask)

        new_residues[dec_mask] = (residues[dec_mask] - 1) % p_broadcast[dec_mask]
        new_residues[inc_mask] = (residues[inc_mask] + 1) % p_broadcast[inc_mask]

        return new_residues

    def get_state(self) -> Dict[str, Any]:
        """Returns the serialized state of the Multi-Scale RNS Field."""
        decoded_macro = self.macro_rns.decode(self.macro_residues)
        return {
            "macro_decoded": decoded_macro.tolist(),
            "macro_energy": self.macro_energy.tolist(),
            "macro_potential": self.get_macro_potential().tolist(),
            "active_micro_grids": {
                f"{y},{x}": {
                    "decoded": mgrid.rns.decode(mgrid.residues).tolist(),
                    "energy": mgrid.energy.tolist(),
                    "potential": self.get_micro_potential(mgrid).tolist()
                } for (y, x), mgrid in self.micro_grids.items()
            }
        }
