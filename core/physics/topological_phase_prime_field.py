import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Precomputed first 100 non-trivial Riemann zeta zero imaginary parts (gamma_n)
# These act as stable intrinsic frequency modes of the base field.
RIEMANN_ZEROS = np.array([
    14.13472514, 21.02203964, 25.01085758, 30.42487612, 32.93506159,
    37.58617816, 40.91871901, 43.32707328, 48.00515088, 49.77383248,
    52.97032148, 56.44624770, 59.34704400, 60.83177852, 65.11254405,
    67.07981053, 69.54640171, 72.06715767, 75.70469070, 77.14484007,
    79.33737502, 82.91038085, 84.73549294, 87.42527461, 88.90941688,
    92.49189927, 95.87063423, 98.83119422, 101.41785029, 103.72553804,
    105.44662305, 107.16861118, 111.02953554, 111.87465918, 114.32022091,
    116.22668032, 118.79072867, 121.37012500, 122.94672941, 124.25681859,
    127.51668388, 129.57870420, 131.01272014, 133.47253228, 134.75650975,
    138.11604205, 139.73620895, 141.11971740, 143.11184581, 146.00098249,
    147.42276534, 150.05352048, 150.92525799, 153.02469381, 156.12429373,
    157.59759182, 158.84990267, 161.18896414, 163.03071624, 165.97476934,
    167.15934009, 169.23178292, 172.29656461, 173.13654406, 174.75700732,
    177.41113645, 179.11666873, 180.34444535, 182.20722101, 184.81423450,
    185.80784964, 187.46914597, 189.41615867, 192.02581691, 193.07689104,
    195.26532431, 196.88456637, 198.01525988, 201.26475194, 202.48628045,
    204.18967180, 205.39462107, 207.90621453, 209.53124808, 211.69085810,
    213.34790317, 214.54511528, 216.16252917, 219.06752009, 220.71490989,
    221.43128189, 224.00700025, 224.98332483, 227.02058319, 229.33633857,
    231.25018742, 231.99615562, 233.64069814, 236.52422967, 237.76864147
], dtype=np.float32)

class TopologicalPhasePrimeField:
    """
    [Topological Phase-Prime Field Model]
    A continuous phase field governed by the superposition of Riemann zeros
    acting as eigenfrequencies, enabling explainable intention/choice emergence.

    Adheres to the "4 Continuities" of Elysia's Causal Field.
    """
    def __init__(self, num_modes: int = 100, min_u: float = 0.1, max_u: float = 5.0, steps_u: int = 500):
        self.num_modes = min(num_modes, len(RIEMANN_ZEROS))
        self.gammas = RIEMANN_ZEROS[:self.num_modes]
        self.min_u = min_u
        self.max_u = max_u
        self.steps_u = steps_u
        self.u_grid = np.linspace(min_u, max_u, steps_u, dtype=np.float32)

        # Dynamic metacognitive tension parameters
        self.sigma = 0.5  # Crucial Line parameter (baseline)
        self.epsilon = 0.0 # Leakage variance

    def set_metacognitive_tension(self, sigma: float = 0.5, epsilon: float = 0.0):
        """
        Dynamically adjusts the metacognitive tension boundary.
        sigma = 0.5 -> Ideal baseline (perfect symmetry, RH holds)
        epsilon > 0.0 -> Dynamic leakage, creative instability/fluctuations
        """
        self.sigma = float(sigma)
        self.epsilon = float(epsilon)

    def compute_field(self, ext_stimulus_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """
        [Continuous Phase Field Phi(u) calculation]
        phi(u) = \\sum_{n} e^{- (sigma + i * gamma_n) * u} + ExtStimulus(u)

        Returns complex wave values across u_grid.
        """
        # Complex decay factor for each u
        # (sigma + epsilon) is the effective real component
        effective_sigma = self.sigma + self.epsilon

        # Grid computation using broadcasting for efficiency
        # gammas shape: (M,), u_grid shape: (U,)
        # exponent shape: (M, U)
        exponent = - (effective_sigma + 1j * self.gammas[:, np.newaxis]) * self.u_grid[np.newaxis, :]
        phi = np.sum(np.exp(exponent), axis=0) # sum over modes, shape: (U,)

        if ext_stimulus_wave is not None:
            # Add continuous external environmental sensory feedback
            phi += ext_stimulus_wave

        return phi

    def compute_spatial_curvature(self, phi: np.ndarray) -> np.ndarray:
        """
        [Spatial Curvature Field K(u)]
        Measures the second derivative/local tension of the phase field.
        Nodes of constructive resonance map to prime coordinates u = ln p.
        """
        # Take the real part representing active physical/informational density
        rho_phase = np.real(phi)

        # Numerical second-derivative to capture phase curvature
        du = (self.max_u - self.min_u) / (self.steps_u - 1)
        k_u = np.gradient(np.gradient(rho_phase, du), du)

        # Normalize or regularize K(u) so background flattening and peak-popping are clear
        # Background cancellation maps towards negative friction, peak resonances soar positive
        return k_u

    def decode_active_prime_nodes(self, k_u: np.ndarray, threshold_mult: float = 1.5) -> List[Tuple[float, int, float]]:
        """
        Extracts active prime nodes (intents) where spatial curvature spikes.
        Returns a list of tuples: (u_peak, nearest_prime, peak_intensity)
        """
        peaks: List[Tuple[float, int, float]] = []

        # Local maxima detection on curvature field
        mean_k = np.mean(k_u)
        std_k = np.std(k_u)
        threshold = mean_k + threshold_mult * std_k

        for i in range(1, len(k_u) - 1):
            if k_u[i] > k_u[i-1] and k_u[i] > k_u[i+1] and k_u[i] > threshold:
                u_peak = self.u_grid[i]
                # Map logarithmic u back to the physical number scale
                x_val = np.exp(u_peak)
                nearest_p = int(round(x_val))

                # Check if it corresponds to an actual prime node resonance
                # (Simple primality check for local physical mapping)
                if self._is_prime(nearest_p):
                    peaks.append((float(u_peak), nearest_p, float(k_u[i])))

        return sorted(peaks, key=lambda x: x[2], reverse=True)

    def _is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
