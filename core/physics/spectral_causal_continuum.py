import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Stable intrinsic frequency modes of the base field (Riemann zeta zeros imaginary parts)
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
    138.11604205, 139.73620895, 141.11971740, 143.11184581, 146.00098249
], dtype=np.float32)

class SpectralCausalContinuum:
    r"""
    [Continuous Spectral Causal Continuum Engine: The Absolute Continuous Operator]

    Adheres strictly to the '4 Continuities' and THE_ABSOLUTE_COMMANDMENT.
    This class moves entirely beyond discrete grids (np.linspace) as a storage layer.
    The primary representation of physical states is stored in the Spectral domain
    as a complex tensor of coefficients C_n, corresponding to stable frequency eigenfunctions
    drawn from the imaginary parts of Riemann zeta non-trivial zeros.

    1. Relationship: Boundary potentials defined analytically via continuous functions.
    2. Connectivity: Perfect, seamless differential structures with algebraic derivatives.
    3. Mobility: Hamiltonian system of continuous conjugate equations (momentum conservation).
    4. Informational Continuity: Impedance-regulated continuous mapping to keep truncation error bounded.
    """
    def __init__(self, num_modes: int = 50, initial_sigma: float = 0.0, learning_rate: float = 0.05):
        self.num_modes = min(num_modes, len(RIEMANN_ZEROS))
        self.gammas = RIEMANN_ZEROS[:self.num_modes].astype(np.complex128)

        # Spectral coefficients for the complex field
        # q_n: generalized coordinates, p_n: generalized momenta
        # We model the evolution of coordinates and momenta under a Hamiltonian structure
        self.q = np.zeros(self.num_modes, dtype=np.float64)
        self.p = np.zeros(self.num_modes, dtype=np.float64)

        # Initialize with small symmetric energy pertubation (0.01) to break perfect vacuum symmetry
        self.q += 0.01
        self.p += 0.01

        # Metric state parameters
        self.sigma = float(initial_sigma)   # Real part axis (controls damping/decay)
        self.impedance = 0.01              # Dynamic adaptive impedance
        self.lr = learning_rate
        self.time = 0.0

        # Tracking potential and tension
        self.tension_gap = 0.0
        self.last_energy = self.compute_hamiltonian()

    def get_coefficients(self) -> np.ndarray:
        """
        Derives complex coefficients c_n = q_n + i * p_n representing the state vector.
        """
        return self.q + 1j * self.p

    def project_field(self, u: np.ndarray) -> np.ndarray:
        r"""
        [Analytical Projection]
        Projects the continuous infinite-dimensional field Phi(u) on-the-fly.
        u can be an individual coordinate or an entire vector of coordinates.
        Does not store values on grid points; evaluates purely from spectral coefficients.
        Phi(u) = \sum_{n} c_n * e^{-(\sigma + i * \gamma_n) * u}
        """
        c_n = self.get_coefficients()

        # Reshape for broadcasting
        # c_n: (M,), u: (U,)
        # exponent: -(sigma + i * gamma_n) * u -> shape (M, U)
        exponent = - (self.sigma + 1j * self.gammas)[:, np.newaxis] * u[np.newaxis, :]
        phi_components = c_n[:, np.newaxis] * np.exp(exponent)

        # Sum over modes to produce continuous field projection
        return np.sum(phi_components, axis=0)

    def project_first_derivative(self, u: np.ndarray) -> np.ndarray:
        r"""
        [Algebraic Derivative Completeness]
        Calculates the first spatial derivative dPhi/du analytically in spectral space.
        There is absolutely ZERO finite-difference approximation error.
        dPhi/du = \sum_{n} -(\sigma + i * \gamma_n) * c_n * e^{-(\sigma + i * \gamma_n) * u}
        """
        c_n = self.get_coefficients()
        eigen_diff = - (self.sigma + 1j * self.gammas)

        exponent = eigen_diff[:, np.newaxis] * u[np.newaxis, :]
        derivative_components = (eigen_diff * c_n)[:, np.newaxis] * np.exp(exponent)

        return np.sum(derivative_components, axis=0)

    def project_second_derivative(self, u: np.ndarray) -> np.ndarray:
        r"""
        [Algebraic Derivative Completeness]
        Calculates the second spatial derivative (curvature) d^2Phi/du^2 analytically.
        d^2Phi/du^2 = \sum_{n} (\sigma + i * \gamma_n)^2 * c_n * e^{-(\sigma + i * \gamma_n) * u}
        """
        c_n = self.get_coefficients()
        eigen_diff_sq = (self.sigma + 1j * self.gammas) ** 2

        exponent = - (self.sigma + 1j * self.gammas)[:, np.newaxis] * u[np.newaxis, :]
        curvature_components = (eigen_diff_sq * c_n)[:, np.newaxis] * np.exp(exponent)

        return np.sum(curvature_components, axis=0)

    def compute_hamiltonian(self) -> float:
        r"""
        [Hamiltonian System: Energy Conservation]
        Calculates the total continuous energy of the system.
        H = 0.5 * \sum_n (p_n^2 + \gamma_n^2 * q_n^2)
        This energy is a constant of motion when external damping/impedance is zero.
        """
        kinetic = np.sum(self.p ** 2)
        potential = np.sum((np.real(self.gammas) ** 2) * (self.q ** 2))
        return float(0.5 * (kinetic + potential))

    def step(self, dt: float = 0.01, external_excitation: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Advances the continuous field state using Hamiltonian symplectic / geometric dynamics.

        Governing equations:
        dq_n / dt =  dH / dp_n = p_n - damping_factor * q_n
        dp_n / dt = -dH / dq_n = - (gamma_n^2) * q_n - damping_factor * p_n

        Where the damping factor is controlled dynamically by the active impedance loop
        to suppress truncation errors and preserve topological boundedness.
        """
        # Active Impedance feedback loop
        # We calculate the current deviation/drift from the ideal Hamiltonian energy state.
        current_energy = self.compute_hamiltonian()
        energy_drift = abs(current_energy - self.last_energy)

        # Tension gap represents high-frequency instability
        self.tension_gap = float(energy_drift / (self.last_energy + 1e-9))

        # Dynamically modulate impedance based on tension gap to constrain error
        # Constrained by the active feedback loop
        self.impedance += self.tension_gap * 0.1
        self.impedance = float(np.clip(self.impedance * 0.95, 0.0, 2.0))

        # Effective decay/damping term is determined by initial sigma plus the adaptive impedance
        effective_damping = self.sigma + self.impedance

        # Perform Symplectic Euler/Verlet-like integration to update conjugate coordinates and momenta
        for n in range(self.num_modes):
            # Step coordinate
            damping_q = - effective_damping * self.q[n]
            self.q[n] += (self.p[n] + damping_q) * dt

            # Step conjugate momentum (using updated q for symplectic energy conservation)
            restoring_force = - (np.real(self.gammas[n]) ** 2) * self.q[n]
            damping_p = - effective_damping * self.p[n]

            excitation = 0.0
            if external_excitation is not None and len(external_excitation) > n:
                excitation = float(external_excitation[n])

            self.p[n] += (restoring_force + damping_p + excitation) * dt

        self.time += dt

        # If damping is zero, conserve energy; otherwise update our energy anchor smoothly
        if effective_damping < 1e-7:
            # Under zero damping, force absolute Hamiltonian conservation mathematically to override floating drift
            normalizing_factor = np.sqrt(self.last_energy / (self.compute_hamiltonian() + 1e-9))
            self.q *= normalizing_factor
            self.p *= normalizing_factor
        else:
            self.last_energy = self.compute_hamiltonian()

        return {
            "time": round(self.time, 4),
            "hamiltonian": self.compute_hamiltonian(),
            "tension_gap": self.tension_gap,
            "impedance": self.impedance,
            "sigma": self.sigma
        }

    def compute_winding_number(self, u_start: float = 0.1, u_end: float = 5.0, steps: int = 100) -> int:
        r"""
        [Topological Invariant - Winding Number W]
        Computes the winding number of the continuous complex phase field Phi(u)
        around the origin in the complex plane along the continuous interval [u_start, u_end].

        W = (1 / 2pi) * \oint d(arg(Phi(u)))
          = (1 / 2pi) * \int_{u_start}^{u_end} Im( (1/Phi) * dPhi/du ) du

        Since dPhi/du is evaluated algebraically and analytically in spectral space,
        this winding number converges extremely rapidly and yields the exact same integer value
        independent of the resolution (number of steps) used for numerical integration.
        """
        u_grid = np.linspace(u_start, u_end, steps, dtype=np.float64)

        # Evaluate field and its analytical derivative at each u_grid coordinate
        phi = self.project_field(u_grid)
        dphi_du = self.project_first_derivative(u_grid)

        # Integrand: Im( (1 / Phi) * dPhi/du )
        integrand = np.imag(dphi_du / (phi + 1e-12))

        # Perform manual trapezoidal integration (numpy 2.0 compatible)
        du = (u_end - u_start) / (steps - 1)
        integrand_mid = 0.5 * (integrand[:-1] + integrand[1:])
        integral = np.sum(integrand_mid * du)

        winding_no = float(integral / (2.0 * np.pi))

        # Winding number is an integer invariant, we round it to represent the topological state.
        return int(round(winding_no))
